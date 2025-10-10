#!/bin/bash

docker build -f reward_utils/Dockerfile -t design2code .

echo "Running prepare_original_info.py in Docker container..."
CONTAINER_ID=$(docker run -d design2code python prepare_original_info.py)
echo "Waiting for docker to complete..."
docker wait $CONTAINER_ID

# Check if the test was successful
EXIT_CODE=$(docker inspect $CONTAINER_ID --format='{{.State.ExitCode}}')

if [ $EXIT_CODE -eq 0 ]; then
    # Copy files from the container
    echo "Copying files from container..."
    # First, copy the entire test_output directory
    docker cp $CONTAINER_ID:/sandbox/testset_final ./test_output_from_container
    # Then move the contents to our test_output directory
    if [ -d "./test_output_from_container" ]; then
        cp -r ./test_output_from_container/* testset_final/
        rm -rf ./test_output_from_container
    else
        # Fallback to original method
        docker cp $CONTAINER_ID:/sandbox/testset_final/. testset_final/
    fi
    echo "Prep completed!"
else
    echo "❌ Prepare failed with exit code: $EXIT_CODE"
    echo "Container logs:"
    docker logs $CONTAINER_ID
fi

# Clean up the container
docker rm $CONTAINER_ID > /dev/null 2>&1 