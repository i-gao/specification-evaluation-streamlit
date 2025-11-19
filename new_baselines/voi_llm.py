from typing import Dict, Tuple, Optional, List
import json
import torch
import sys
import os

# Add voi directory to path to import VOI modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'voi'))
from voi import VOIAlgorithm
from environment import SimulationConfig

from new_baselines.single_llm import SingleLLM
from utils.misc import (
    print_debug,
    parse_json,
)


class VOILLM(SingleLLM):
    """
    Agent that uses Value of Information (VOI) algorithm to make principled decisions
    about which options to show and which features to highlight.
    
    Key approach:
    1. Generate candidate solutions (phi matrix) and feature space upfront
    2. Maintain Gaussian posterior over user's theta (preference weights)
    3. Each turn: VOI algorithm decides O (which solutions to show) and Q (which features to highlight)
    4. Update posterior based on user's revealed preferences
    """

    def __init__(
        self,
        *args,
        gamma_Y: float = 1.0,  # Cost exponent for number of solutions shown
        gamma_F: float = 1.0,  # Cost exponent for number of features highlighted
        lam: float = 1e-2,     # Regularization parameter for posterior updates
        mc_samples_theta: int = 20,  # Monte Carlo samples for advantage computation
        mc_samples_S: int = 20,      # Monte Carlo samples for spontaneous discovery
        n_highlight: int = 2,         # Number of features to highlight
        n_options: int = 3,           # Number of options to show (for strategy C)
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Hierarchical planning state
        self.subtasks: Optional[List[Dict]] = None  # List of subtask dicts
        self.current_subtask_idx: int = 0
        self.current_subtask: Optional[Dict] = None
        
        # VOI hyperparameters
        self.gamma_Y = gamma_Y
        self.gamma_F = gamma_F
        self.lam = lam
        self.mc_samples_theta = mc_samples_theta
        self.mc_samples_S = mc_samples_S
        self.n_highlight = n_highlight
        self.n_options = n_options
        
        # Feature space and solution space (scoped to current subtask)
        self.feature_names: Optional[List[str]] = None  # Ordered list of feature names
        self.d: Optional[int] = None  # Number of features
        self.candidate_solutions: Optional[List[Dict]] = None  # List of solution dicts
        self.phi: Optional[torch.Tensor] = None  # (n_y, d) feature matrix
        
        # VOI algorithm instance (scoped to current subtask)
        self.voi_algorithm: Optional[VOIAlgorithm] = None
        
        # Register hooks
        self._pre_conversation_hooks = list(
            getattr(self, "_pre_conversation_hooks", [])
        ) + [
            self._hook_hierarchical_planning,
            self._hook_generate_solution_space,
            self._hook_initialize_voi,
        ]
        
        self._post_user_response_hooks = list(
            getattr(self, "_post_user_response_hooks", [])
        ) + [
            self._insert_system_message,
            self._insert_user_message,
            self._hook_update_voi_posterior,
        ]

    ############ HOOKS ############

    def _insert_system_message(self, hook_state: Dict) -> None:
        """Insert the system message on first turn."""
        if not self.has_seen_system_prompt:
            system_msg = self._get_generate_prompt()
            self.agent_executor.insert_message("system", system_msg)
            self.has_seen_system_prompt = True

    def _insert_user_message(self, hook_state: Dict) -> None:
        """Insert the most recent user message."""
        conversation_history = hook_state.get("conversation_history", []) or []
        last_user_msg = None
        if len(conversation_history) > 0 and getattr(
            conversation_history[-1], "user_msg", None
        ):
            last_user_msg = conversation_history[-1].user_msg
        if last_user_msg is None:
            last_user_msg = ""
        self.agent_executor.insert_message("user", last_user_msg)

    def _hook_hierarchical_planning(self, hook_state: Dict) -> Dict:
        """
        Decompose the task into subtasks. Each subtask will have its own
        solution space (phi matrix) and VOI algorithm instance.
        """
        initial_user_msg = None
        conversation_history = hook_state.get("conversation_history", []) or []
        if len(conversation_history) > 0 and getattr(
            conversation_history[-1], "user_msg", None
        ):
            initial_user_msg = conversation_history[-1].user_msg
        if initial_user_msg is None:
            initial_user_msg = ""

        prompt = (
            "system",
            f"""You are creating a hierarchical plan for completing a task.

Task context:
{self.commonsense_instructions}
User request: {initial_user_msg}

Break the task into mostly independent subtasks that can be solved one at a time.
Each subtask should be a meaningful sub-problem with its own decision space.

Good example (plan trip to Paris):
1. Find flight - needs: dates, budget. Outputs: selected flight.
2. Find hotel - needs: dates, location preference. Outputs: selected hotel.
3. Find restaurants - needs: cuisine preferences. Outputs: selected restaurants.

Bad example (find 2 hoodies):
Single task - not meaningfully decomposable, treat as one subtask.

If not decomposable, create a single subtask for the entire task.

Use tools to understand task space, then respond with JSON array of subtasks:
[
  {{"name": "subtask_name", "description": "what needs to be decided", "dependencies": []}},
  ...
]

If single task: [{{"name": "complete_task", "description": "{initial_user_msg}", "dependencies": []}}]
""",
        )

        raw, _, _ = self._call_agent_executor(
            prompt, persist_state=False, min_react_steps=3 if len(self.actions) > 0 else 1
        )

        content = raw.strip() if raw is not None else "[]"
        subtasks = parse_json(content) or []
        if not isinstance(subtasks, list) or len(subtasks) == 0:
            # Fallback: single subtask
            subtasks = [{"name": "complete_task", "description": initial_user_msg, "dependencies": []}]
        
        self.subtasks = subtasks
        self.current_subtask_idx = 0
        self.current_subtask = subtasks[0] if subtasks else None

        if self.verbosity:
            print_debug(
                f"Hierarchical plan: {len(subtasks)} subtasks",
                "hierarchical_planning",
                color="orange",
            )
        return {"subtasks": subtasks}

    def _hook_generate_solution_space(self, hook_state: Dict) -> Dict:
        """
        Generate the solution space FOR THE CURRENT SUBTASK: 
        1. Candidate solutions (shortlist of relevant options)
        2. Feature space (ordered list of features)
        3. Feature matrix phi (numeric representation of each solution)
        """
        initial_user_msg = None
        conversation_history = hook_state.get("conversation_history", []) or []
        if len(conversation_history) > 0 and getattr(
            conversation_history[-1], "user_msg", None
        ):
            initial_user_msg = conversation_history[-1].user_msg
        if initial_user_msg is None:
            initial_user_msg = ""

        # Scope to current subtask
        subtask_context = ""
        if self.current_subtask:
            subtask_context = f"\nCurrent subtask: {self.current_subtask.get('name')} - {self.current_subtask.get('description')}"

        # Step 1: Generate candidate solutions FOR CURRENT SUBTASK
        prompt_solutions = (
            "system",
            f"""You are preparing a shortlist of candidate solutions FOR THE CURRENT SUBTASK.

Task context:
{self.commonsense_instructions}
User request: {initial_user_msg}
{subtask_context}

Use the available tools to create a shortlist of 15-30 candidate solutions RELEVANT TO THIS SUBTASK. These should be diverse and cover different parts of the solution space for this subtask.

For example, if the subtask is "find flight", retrieve flights. If the subtask is "find hotel", retrieve hotels.

Respond with JSON array of solution objects. Each solution should have:
- "id": unique identifier (string or number)
- "name": human-readable name
- Any other relevant attributes from the data

Example:
[
  {{"id": "123", "name": "Blue Cotton Hoodie", "price": 45, "brand": "Nike", ...}},
  {{"id": "456", "name": "Black Fleece Hoodie", "price": 60, "brand": "Adidas", ...}},
  ...
]
""",
        )

        raw_solutions, _, _ = self._call_agent_executor(
            prompt_solutions, persist_state=False, min_react_steps=3 if len(self.actions) > 0 else 1
        )

        candidate_solutions = parse_json(raw_solutions.strip() if raw_solutions else "[]") or []
        if not isinstance(candidate_solutions, list):
            candidate_solutions = []
        
        self.candidate_solutions = candidate_solutions
        n_y = len(candidate_solutions)

        if self.verbosity:
            print_debug(
                f"Generated {n_y} candidate solutions",
                "generate_solution_space",
                color="orange",
            )

        # Step 2: Generate feature space FOR CURRENT SUBTASK
        prompt_features = (
            "system",
            f"""You are defining the feature space for the CURRENT SUBTASK.

Task context:
{self.commonsense_instructions}
{subtask_context}
Candidate solutions: {len(candidate_solutions)} items

Generate an ordered list of 20-30 decision-relevant features that describe the candidate solutions FOR THIS SUBTASK. These features should:
1. Be relevant to user preferences for this subtask
2. Vary across the candidate solutions (discriminative)
3. Cover both explicit attributes (price, color) and implicit qualities (style, comfort)

Respond with JSON array of feature names (strings):
["feature_1", "feature_2", ...]

Example for clothing:
["price", "brand_reputation", "material_quality", "style_modernity", "color_versatility", "comfort_rating", "durability", "fit_type", ...]
""",
        )

        raw_features, _, _ = self._call_agent_executor(
            prompt_features, persist_state=False, min_react_steps=1
        )

        feature_names = parse_json(raw_features.strip() if raw_features else "[]") or []
        if not isinstance(feature_names, list):
            feature_names = []
        
        feature_names = [str(f).strip() for f in feature_names if f]
        self.feature_names = feature_names
        self.d = len(feature_names)

        if self.verbosity:
            print_debug(
                f"Generated {self.d} features: {feature_names[:5]}...",
                "generate_solution_space",
                color="orange",
            )

        # Step 3: Generate phi (feature matrix)
        solutions_json = json.dumps(candidate_solutions, indent=2)
        features_json = json.dumps(feature_names, indent=2)
        
        prompt_phi = (
            "system",
            f"""You are encoding candidate solutions as feature vectors.

Features (in order): {features_json}

Candidate solutions: {solutions_json}

For each solution, provide a numeric feature vector where each value represents the solution's value on that feature dimension. Use a scale appropriate for each feature (e.g., price in dollars, ratings 0-10, binary features 0/1, etc.).

Respond with JSON object mapping solution_id -> feature_vector:
{{
  "solution_id": [value_for_feature_1, value_for_feature_2, ...],
  ...
}}

Ensure the feature vectors are in the same order as the feature list.
""",
        )

        raw_phi, _, _ = self._call_agent_executor(
            prompt_phi, persist_state=False, min_react_steps=1
        )

        phi_dict = parse_json(raw_phi.strip() if raw_phi else "{}") or {}
        if not isinstance(phi_dict, dict):
            phi_dict = {}

        # Convert to tensor
        phi_list = []
        for sol in candidate_solutions:
            sol_id = str(sol.get("id", ""))
            if sol_id in phi_dict:
                vec = phi_dict[sol_id]
                if isinstance(vec, list) and len(vec) == self.d:
                    phi_list.append(vec)
                else:
                    # Fallback: random vector
                    phi_list.append([0.0] * self.d)
            else:
                # Fallback: random vector
                phi_list.append([0.0] * self.d)

        self.phi = torch.tensor(phi_list, dtype=torch.float32)  # (n_y, d)

        if self.verbosity:
            print_debug(
                f"Generated phi matrix: {self.phi.shape}",
                "generate_solution_space",
                color="orange",
            )

        return {
            "candidate_solutions": candidate_solutions,
            "feature_names": feature_names,
            "phi": self.phi,
        }

    def _hook_initialize_voi(self, hook_state: Dict) -> Dict:
        """
        Initialize the VOI algorithm with a prior over theta.
        """
        initial_user_msg = None
        conversation_history = hook_state.get("conversation_history", []) or []
        if len(conversation_history) > 0 and getattr(
            conversation_history[-1], "user_msg", None
        ):
            initial_user_msg = conversation_history[-1].user_msg
        if initial_user_msg is None:
            initial_user_msg = ""

        features_json = json.dumps(self.feature_names, indent=2)

        prompt_prior = (
            "system",
            f"""You are setting a prior distribution over user preferences for this task.

Task context:
{self.commonsense_instructions}
User request: {initial_user_msg}
Features: {features_json}

For each feature, estimate the user's likely preference weight (theta). Positive values mean the user prefers higher values of that feature, negative means they prefer lower values.

Also estimate your uncertainty (standard deviation) for each weight.

Respond with JSON object:
{{
  "mu": [mean_1, mean_2, ...],  // Prior mean for each feature (length {self.d})
  "sigma_diag": [std_1, std_2, ...]  // Prior std dev for each feature (length {self.d})
}}

Example: If users typically prefer lower prices and higher quality:
{{
  "mu": [-5.0, 8.0, ...],  // negative for price (prefer lower), positive for quality (prefer higher)
  "sigma_diag": [2.0, 3.0, ...]  // uncertainty
}}
""",
        )

        raw_prior, _, _ = self._call_agent_executor(
            prompt_prior, persist_state=False, min_react_steps=1
        )

        prior_dict = parse_json(raw_prior.strip() if raw_prior else "{}") or {}
        
        # Extract mu and Sigma
        mu_list = prior_dict.get("mu", [0.0] * self.d)
        sigma_diag = prior_dict.get("sigma_diag", [1.0] * self.d)
        
        # Ensure correct length
        if len(mu_list) != self.d:
            mu_list = [0.0] * self.d
        if len(sigma_diag) != self.d:
            sigma_diag = [1.0] * self.d

        mu0 = torch.tensor(mu_list, dtype=torch.float32)
        Sigma0 = torch.diag(torch.tensor(sigma_diag, dtype=torch.float32) ** 2)

        # Create SimulationConfig
        cfg = SimulationConfig(
            d=self.d,
            gamma_Y=self.gamma_Y,
            gamma_F=self.gamma_F,
            C_budget=self.interaction_budget,
            theta_true=mu0,  # Won't be used directly, just for structure
            phi=self.phi,
            lam=self.lam,
        )

        # Initialize VOI algorithm
        self.voi_algorithm = VOIAlgorithm(
            cfg=cfg,
            mu0=mu0,
            Sigma0=Sigma0,
            n_highlight=self.n_highlight,
            n_options=self.n_options,
            mc_samples_theta=self.mc_samples_theta,
            mc_samples_S=self.mc_samples_S,
            initial_features=set(),  # Start with no features discovered
        )

        if self.verbosity:
            print_debug(
                f"Initialized VOI with prior mu: {mu0[:5]}..., Sigma_diag: {torch.diag(Sigma0)[:5]}...",
                "initialize_voi",
                color="orange",
            )

        return {"mu0": mu0, "Sigma0": Sigma0}

    def _hook_update_voi_posterior(self, hook_state: Dict) -> Dict:
        """
        Parse user response to extract revealed preferences and update VOI posterior.
        """
        conversation_history = hook_state.get("conversation_history", []) or []
        last_user_msg = ""
        if len(conversation_history) > 0 and getattr(
            conversation_history[-1], "user_msg", None
        ):
            last_user_msg = conversation_history[-1].user_msg

        features_json = json.dumps(self.feature_names, indent=2)

        prompt = (
            "system",
            f"""Analyze the user's message to identify which features they mentioned or expressed preferences about.

Current feature space: {features_json}

User message: {last_user_msg}

Identify:
1. Which features (by index) the user explicitly mentioned or implicitly revealed preferences about
2. For each revealed feature, estimate the user's preference direction and strength on a scale

Respond with JSON:
{{
  "discovered_feature_indices": [0, 5, 12, ...],  // Indices of features user revealed preferences about
  "preference_hints": {{  // Optional: hints about user's theta values
    "0": 5.0,  // User seems to prefer higher values of feature 0
    "5": -3.0,  // User seems to prefer lower values of feature 5
    ...
  }}
}}
""",
        )

        raw, _, _ = self._call_agent_executor(
            prompt, persist_state=False, min_react_steps=1
        )

        parsed = parse_json(raw.strip() if raw else "{}") or {}
        discovered_indices = parsed.get("discovered_feature_indices", [])
        if not isinstance(discovered_indices, list):
            discovered_indices = []
        
        discovered_features = set(int(i) for i in discovered_indices if isinstance(i, (int, float)))

        # For now, use the current posterior mean as the "observed" theta (with discovered features marked)
        # In a real setting, we'd extract actual preference signals from user's response
        theta_projected = self.voi_algorithm.posterior.mean.clone()
        
        # Update VOI algorithm
        if len(discovered_features) > 0:
            self.voi_algorithm.update(discovered_features, theta_projected)

        if self.verbosity:
            print_debug(
                f"Updated VOI posterior with discovered features: {discovered_features}",
                "update_voi_posterior",
                color="orange",
            )

        return {"discovered_features": discovered_features}

    def _check_subtask_completion(self, user_response: str) -> bool:
        """Check if current subtask is complete and transition to next if so."""
        if not self.subtasks or len(self.subtasks) <= 1:
            return False  # No subtasks or single subtask
        
        if self.current_subtask_idx >= len(self.subtasks) - 1:
            return False  # Already on last subtask
        
        # Ask model if subtask is complete
        prompt = (
            "system",
            f"""Determine if the current subtask is complete based on the user's most recent response.

Current subtask: {self.current_subtask.get('name')} - {self.current_subtask.get('description')}
User's response: {user_response}

Respond with JSON:
{{"complete": true/false, "reason": "explanation"}}
""",
        )
        
        raw, _, _ = self._call_agent_executor(prompt, persist_state=False, min_react_steps=1)
        result = parse_json(raw.strip() if raw else "{}") or {}
        
        if result.get("complete", False):
            # Move to next subtask
            self.current_subtask_idx += 1
            self.current_subtask = self.subtasks[self.current_subtask_idx]
            
            # Re-generate solution space and VOI for new subtask
            self._hook_generate_solution_space({})
            self._hook_initialize_voi({})
            
            if self.verbosity:
                print_debug(
                    f"Transitioning to subtask {self.current_subtask_idx + 1}/{len(self.subtasks)}: {self.current_subtask.get('name')}",
                    "subtask_transition",
                    color="yellow",
                )
            return True
        
        return False

    ############ PROMPTS ############

    def _get_generate_prompt(self) -> str:
        """System prompt for generating messages."""
        # Add current subtask context
        current_subtask_section = ""
        if self.current_subtask and self.subtasks and len(self.subtasks) > 1:
            subtask_list = "\n".join([
                f"{i+1}. {st.get('name')}: {st.get('description')}"
                for i, st in enumerate(self.subtasks)
            ])
            current_subtask_section = f"""

Hierarchical Plan - Currently on Subtask {self.current_subtask_idx + 1}/{len(self.subtasks)}:
{subtask_list}

Current subtask: {self.current_subtask.get('name')} - {self.current_subtask.get('description')}
Focus on completing THIS subtask before moving to the next one.
"""

        return f"""You are a helpful assistant using Value of Information principles to help users make decisions.

Task context:
{self.commonsense_instructions}
{current_subtask_section}

You will be provided with guidance on:
- Which candidate solutions to show the user (if any)
- Which features to emphasize in your message

Your goal is to help the user discover their preferences by strategically showing options and highlighting relevant features.

Guidelines:
- Follow the VOI algorithm's recommendations for which solutions to show and which features to highlight
- When showing options, explicitly mention the highlighted features to draw user's attention
- Work on ONE SUBTASK AT A TIME
- {self.msg_fmt_instructions}

When you have finished the task AND received user confirmation of its completeness, generate the string <END_CONVERSATION>.
To show a user a message, do not make tool calls in that message.
"""

    def generate_message(self, user_response: Optional[str] = None) -> Tuple[str, bool]:
        """
        Generate next message using VOI algorithm's recommendations.
        """
        # Check if current subtask is complete and transition if needed
        if user_response:
            self._check_subtask_completion(user_response)
        
        if self.verbosity == 2:
            print_debug(
                f"Generating message with user_response:\n{user_response}",
                "generate_message",
                color="blue",
            )

        # Run VOI algorithm to get O (solutions to show) and Q (features to highlight)
        O_tensor, Q_set = self.voi_algorithm.step()  # O: (k, d), Q: set of feature indices
        
        # Convert tensor indices to actual solutions
        if len(O_tensor) > 0:
            # Find which solution indices these correspond to
            # O_tensor contains feature vectors, need to match them to phi rows
            solution_indices = []
            for o_vec in O_tensor:
                # Find closest match in phi (exact match expected)
                dists = torch.norm(self.phi - o_vec.unsqueeze(0), dim=1)
                closest_idx = torch.argmin(dists).item()
                solution_indices.append(closest_idx)
            
            solutions_to_show = [self.candidate_solutions[i] for i in solution_indices]
        else:
            solutions_to_show = []

        # Get feature names to highlight
        features_to_highlight = [self.feature_names[i] for i in Q_set]

        # Insert VOI guidance as internal thought
        voi_guidance = {
            "solutions_to_show": solutions_to_show,
            "features_to_highlight": features_to_highlight,
            "strategy": "A" if len(O_tensor) == 0 else "B" if len(O_tensor) == 1 else "C",
        }
        
        self.agent_executor.insert_message(
            "assistant",
            f"<think>VOI algorithm guidance:\n"
            f"{json.dumps(voi_guidance, indent=2)}\n"
            f"I should craft my message following this guidance.</think>",
        )

        # Generate message
        raw, _, _ = self._call_agent_executor()
        if raw is None:
            return None, False

        wants_to_end_conversation = "<END_CONVERSATION>" in raw
        assistant_msg = raw.replace("<END_CONVERSATION>", "")

        if self.verbosity:
            print_debug(
                f"Generated message: {raw}",
                "generate_message",
                color="orange",
            )

        return assistant_msg, wants_to_end_conversation

    def get_test_prediction(self) -> str:
        """Get final prediction based on VOI algorithm's best solution."""
        # Get best solution according to VOI
        best_y_ix = self.voi_algorithm.best_y_ix().item()
        best_solution = self.candidate_solutions[best_y_ix]

        prompt = [
            ("system", self._get_predict_prompt()),
            ("user", f"The recommended solution is: {json.dumps(best_solution, indent=2)}")
        ]
        
        if self.verbosity == 2:
            print_debug(
                f"Getting test prediction for best solution: {best_solution.get('name', best_solution.get('id'))}",
                "get_test_prediction",
                color="blue",
            )

        raw, _, _ = self._call_agent_executor(*prompt, persist_state=False)
        if self.verbosity:
            print_debug(
                f"Current prediction: {raw}",
                "get_test_prediction",
                color="orange",
            )

        return raw

