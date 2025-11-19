from typing import List, Dict, Any, Union
from collections import defaultdict


def _parse_dsl_expression(expr: str) -> Dict:
    """
    Parse a DSL expression into the internal condition format.
    
    Format: field_name "value" [OR|AND|NOT] ...
    Supports parentheses for grouping.
    
    Examples:
    - subject_contains "meeting" OR email_id "0"
    - (subject_contains "meeting" AND from_contains "boss") OR email_id "1"
    - subject_contains "meeting" AND NOT content_contains "spam"
    
    Returns:
        Dict representing the condition in internal format
    """
    expr = expr.strip()
    if not expr:
        return {}
    
    # Tokenize: split on operators and parentheses, but preserve them
    # Pattern matches: field_name, quoted strings, operators, parentheses
    tokens = []
    i = 0
    while i < len(expr):
        # Skip whitespace
        if expr[i].isspace():
            i += 1
            continue
        
        # Match quoted strings
        if expr[i] == '"':
            end = expr.find('"', i + 1)
            if end == -1:
                raise ValueError(f"Unclosed quote at position {i}")
            tokens.append(('STRING', expr[i+1:end]))
            i = end + 1
        # Match parentheses
        elif expr[i] in '()':
            tokens.append(('PAREN', expr[i]))
            i += 1
        # Match operators (case-insensitive)
        elif expr[i:i+3].upper() == 'AND':
            tokens.append(('OP', 'AND'))
            i += 3
        elif expr[i:i+2].upper() == 'OR':
            tokens.append(('OP', 'OR'))
            i += 2
        elif expr[i:i+3].upper() == 'NOT':
            tokens.append(('OP', 'NOT'))
            i += 3
        # Match field names (identifier)
        else:
            # Find the end of the identifier
            start = i
            while i < len(expr) and (expr[i].isalnum() or expr[i] == '_'):
                i += 1
            if i > start:
                tokens.append(('FIELD', expr[start:i]))
            else:
                raise ValueError(f"Unexpected character at position {i}: {expr[i]}")
    
    class Parser:
        def __init__(self, tokens):
            self.tokens = tokens
            self.idx = 0
        
        def parse_expression(self):
            """Parse tokens into a condition tree, handling operator precedence."""
            # Parse with operator precedence: NOT > AND > OR
            # First parse all atoms and NOT expressions
            left = self.parse_atom()
            if left is None:
                return None
            
            # Then handle AND (higher precedence)
            while self.idx < len(self.tokens):
                token_type, token_value = self.tokens[self.idx]
                if token_type == 'OP' and token_value == 'AND':
                    self.idx += 1
                    right = self.parse_atom()
                    if right is None:
                        raise ValueError("AND operator requires a right operand")
                    left = {"op": "AND", "items": [left, right]}
                elif token_type == 'OP' and token_value == 'OR':
                    # OR has lower precedence, so we need to parse the rest and combine
                    self.idx += 1
                    right = self.parse_expression()
                    if right is None:
                        raise ValueError("OR operator requires a right operand")
                    left = {"op": "OR", "items": [left, right]}
                    break
                elif token_type == 'PAREN' and token_value == ')':
                    # End of parenthesized expression
                    break
                else:
                    break
            
            return left
        
        def parse_atom(self):
            """Parse an atomic expression (field condition, NOT, or parenthesized expression)."""
            if self.idx >= len(self.tokens):
                return None
            
            token_type, token_value = self.tokens[self.idx]
            
            if token_type == 'PAREN' and token_value == '(':
                # Parse parenthesized expression
                self.idx += 1
                result = self.parse_expression()
                if self.idx >= len(self.tokens) or self.tokens[self.idx] != ('PAREN', ')'):
                    raise ValueError("Unclosed parenthesis")
                self.idx += 1
                return result
            
            elif token_type == 'OP' and token_value == 'NOT':
                # Parse NOT expression
                self.idx += 1
                operand = self.parse_atom()
                if operand is None:
                    raise ValueError("NOT operator requires an operand")
                return {"op": "NOT", "condition": operand}
            
            elif token_type == 'FIELD':
                field = token_value
                self.idx += 1
                if self.idx >= len(self.tokens) or self.tokens[self.idx][0] != 'STRING':
                    raise ValueError(f"Field {field} must be followed by a quoted string")
                value = self.tokens[self.idx][1]
                self.idx += 1
                return {"field": field, "value": [value]}
            
            return None
    
    parser = Parser(tokens)
    result = parser.parse_expression()
    
    if result is None:
        return {}
    return result


def _normalize_text(text: str) -> str:
    """Normalize text for keyword matching: lowercase and remove spaces/dashes/underscores."""
    if not text:
        return ""
    return text.lower().replace("-", "").replace("_", "").replace(" ", "")


def _check_field_condition(email: Dict, field: str, value: Any) -> bool:
    """
    Check if a field condition matches the email.
    
    Args:
        email: The email dictionary
        field: Field name (email_id, subject_contains, from_contains, content_contains)
        value: Value to match (list for contains fields, list for email_id)
    
    Returns:
        True if condition matches
    """
    if field == "email_id":
        email_id = email.get("email_id", "")
        # Convert email_id to string for comparison (email_id can be int or str)
        email_id_str = str(email_id)
        if isinstance(value, list):
            # Convert all values in list to strings for comparison
            value_strs = [str(v) for v in value]
            return email_id_str in value_strs
        return email_id_str == str(value)
    
    elif field == "subject_contains":
        subject = email.get("subject") or ""
        subject_normalized = _normalize_text(subject)
        if isinstance(value, list):
            return any(_normalize_text(keyword) in subject_normalized for keyword in value)
        return _normalize_text(value) in subject_normalized
    
    elif field == "from_contains":
        from_addr = email.get("from") or ""
        from_normalized = _normalize_text(from_addr)
        if isinstance(value, list):
            return any(_normalize_text(keyword) in from_normalized for keyword in value)
        return _normalize_text(value) in from_normalized
    
    elif field == "content_contains":
        message = email.get("message") or ""
        message_normalized = _normalize_text(message)
        if isinstance(value, list):
            return any(_normalize_text(keyword) in message_normalized for keyword in value)
        return _normalize_text(value) in message_normalized
    
    return False


def _evaluate_condition(email: Dict, condition: str) -> bool:
    """
    Evaluate a DSL condition expression against an email.
    
    Args:
        email: The email dictionary
        condition: DSL condition string (e.g., "subject_contains \"meeting\" OR email_id \"0\"")
    
    Returns:
        True if condition matches
    """
    if not isinstance(condition, str):
        return False
    
    # Parse DSL string into internal format
    parsed_condition = _parse_dsl_expression(condition)
    
    # Evaluate the parsed condition
    return _evaluate_parsed_condition(email, parsed_condition)


def _evaluate_parsed_condition(email: Dict, condition: Dict) -> bool:
    """
    Evaluate a parsed condition (internal format) against an email.
    
    Args:
        email: The email dictionary
        condition: Parsed condition dict with op/field/value structure
    
    Returns:
        True if condition matches
    """
    if not isinstance(condition, dict):
        return False
    
    # Check for boolean operators
    if "op" in condition:
        op = condition.get("op")
        
        if op == "AND":
            items = condition.get("items", [])
            return all(_evaluate_parsed_condition(email, item) for item in items)
        
        elif op == "OR":
            items = condition.get("items", [])
            return any(_evaluate_parsed_condition(email, item) for item in items)
        
        elif op == "NOT":
            sub_condition = condition.get("condition")
            if sub_condition is None:
                return False
            return not _evaluate_parsed_condition(email, sub_condition)
        
        else:
            # Unknown operator
            return False
    
    # Check for field condition: {"field": "...", "value": ...}
    if "field" in condition and "value" in condition:
        field = condition.get("field")
        value = condition.get("value")
        return _check_field_condition(email, field, value)
    
    return False


def _is_edge_case_rule(conditions: str) -> bool:
    """
    Check if a rule is an edge case rule (matches only a specific email_id).
    Edge case rules have the format: email_id "X" (with no other conditions).
    
    Args:
        conditions: DSL condition string
    
    Returns:
        True if this is an edge case rule (only matches a specific email_id)
    """
    if not isinstance(conditions, str):
        return False
    
    # Parse the condition to check if it's just an email_id match
    parsed = _parse_dsl_expression(conditions.strip())
    
    # Edge case: condition is just {"field": "email_id", "value": [...]}
    if isinstance(parsed, dict) and "field" in parsed and "op" not in parsed:
        return parsed.get("field") == "email_id"
    
    return False


def apply_email_policy(emails: List[Dict], policy: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Apply rules to organize emails into folders.

    Policy structure (DSL format):
    [
        {
            "conditions": "subject_contains \"meeting\" OR email_id \"0\"",
            "folder": "Meetings"
        },
        {
            "conditions": "(subject_contains \"project\" AND from_contains \"boss\") AND NOT content_contains \"cancelled\"",
            "folder": "Active Projects"
        }
    ]

    Rules are processed in sequence. Once an email matches a rule, it is assigned to that rule's folder
    and won't be matched by subsequent general rules. However, edge case rules (those that match only
    a specific email_id) can override previous matches to ensure they work regardless of what rules
    were applied beforehand.
    """
    working_emails = []
    for e in emails:
        we = e.copy()
        we["_matched"] = False
        we["_folder"] = None
        working_emails.append(we)

    # Ensure policy is a list
    if not isinstance(policy, list):
        policy = []

    # Apply rules in sequence
    for rule in policy:
        conditions = rule.get("conditions", {})
        folder = rule.get("folder", "Unsorted")
        
        # Check if this is an edge case rule (can override previous matches)
        is_edge_case = _is_edge_case_rule(conditions)

        for we in working_emails:
            # Skip already-matched emails for general rules
            # But allow edge case rules to override previous matches
            if we["_matched"] and not is_edge_case:
                continue

            # Evaluate the condition expression
            if _evaluate_condition(we, conditions):
                we["_folder"] = folder
                we["_matched"] = True

    # Output emails organized by folders
    output = defaultdict(list)
    for we in working_emails:
        folder = we.get("_folder") or "Unsorted"
        result_email = {k: v for k, v in we.items() if not k.startswith("_")}
        output[folder].append(result_email)

    return dict(output)


def check_email_folder_assignment(
    policy: List[Dict],
    email_id: Union[str, int],
    correct_folder: str,
    emails_data: List[Dict],
) -> bool:
    """
    Check if an email is assigned to the correct folder.
    Returns True if the email is in the correct folder.
    Uses email_id to uniquely identify the email.
    
    Args:
        policy: List of rule dicts (each with "conditions" and "folder")
        email_id: The email ID to check
        correct_folder: The expected folder name
        emails_data: List of email dictionaries
    """
    try:
        # Ensure policy is a list
        if not isinstance(policy, list):
            return False
        
        organized = apply_email_policy(emails_data, policy)

        # Find which folder the email ended up in
        # Convert email_id to string for comparison (email_id can be int or str)
        email_id_str = str(email_id)
        email_folder = None
        for folder, emails in organized.items():
            for e in emails:
                # Use email_id field to identify the email (compare as strings)
                e_id = e.get("email_id", "")
                if str(e_id) == email_id_str:
                    email_folder = folder
                    break
            if email_folder:
                break

        if email_folder is None:
            return False

        # Compare normalized (case-insensitive, ignoring whitespace differences)
        return email_folder.lower().strip() == correct_folder.lower().strip()
    except Exception:
        return False


def check_rule_satisfaction(
    policy: List[Dict],
    rule_name: str,
    rule_conditions: str,
    rule_folder: str,
    emails_data: List[Dict],
) -> bool:
    """
    Check if a specific rule is satisfied by the policy.
    A rule is satisfied if all emails that match the rule's conditions
    are assigned to the rule's target folder.
    
    Args:
        policy: List of rule dicts (each with "conditions" and "folder")
        rule_name: Name/description of the rule (for error messages)
        rule_conditions: DSL condition string for the rule
        rule_folder: Target folder for the rule
        emails_data: List of email dictionaries
    
    Returns:
        True if all emails matching the rule conditions are in the correct folder
    """
    try:
        if not isinstance(policy, list):
            return False
        
        organized = apply_email_policy(emails_data, policy)
        
        # Find all emails that match the rule conditions
        matching_email_ids = []
        for email in emails_data:
            if _evaluate_condition(email, rule_conditions):
                email_id = str(email.get("email_id", ""))
                if email_id:
                    matching_email_ids.append(email_id)
        
        if not matching_email_ids:
            # Rule matches no emails - consider it satisfied (vacuous truth)
            return True
        
        # Check if all matching emails are in the correct folder
        for email_id in matching_email_ids:
            email_folder = None
            for folder, emails in organized.items():
                for e in emails:
                    if str(e.get("email_id", "")) == email_id:
                        email_folder = folder
                        break
                if email_folder:
                    break
            
            if email_folder is None:
                # Email not assigned to any folder
                return False
            
            # Check if folder matches (case-insensitive)
            if email_folder.lower().strip() != rule_folder.lower().strip():
                return False
        
        return True
    except Exception:
        return False
