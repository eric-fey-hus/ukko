#!/bin/bash

# --- Configuration ---
ZIP_FILENAME_PREFIX="repo_archive" # Prefix for your zip file
TEMP_BRANCH_NAME="tmp_for_zip_export" # Use a more descriptive temp branch name


# --- Script Start ---

echo "Starting repository archival script..."

# 1. Get the current repository root and parent directory
REPO_ROOT=$(git rev-parse --show-toplevel)
if [ $? -ne 0 ]; then
    echo "Error: Not in a Git repository. Exiting."
    exit 1
fi
echo "Repository root: $REPO_ROOT"

REPO_NAME=$(basename "$REPO_ROOT")
PARENT_DIR=$(dirname "$REPO_ROOT")
ZIP_DEST_DIR="$PARENT_DIR"
echo "Zip file will be placed in: $ZIP_DEST_DIR"

# 2. Get the current branch name to return later
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ $? -ne 0 ]; then
    echo "Error: Could not determine current branch. Exiting."
    exit 1
fi
echo "Current branch: $CURRENT_BRANCH"

# 3. Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "Error: You have uncommitted changes. Please commit or stash them before running this script."
    exit 1
fi
echo "No uncommitted changes detected."

# 4. Create and switch to the temporary branch
echo "Creating and switching to branch: $TEMP_BRANCH_NAME"
git checkout -b "$TEMP_BRANCH_NAME"
if [ $? -ne 0 ]; then
    echo "Error: Could not create/switch to temporary branch. Exiting."
    exit 1
fi

# 5. Run nbstripout on all notebooks
echo "Running nbstripout on all notebooks..."
find . -name "*.ipynb" -print0 | xargs -0 -I {} bash -c 'nbstripout "{}" || { echo "Warning: nbstripout failed for {}"; exit 1; }'
if [ $? -ne 0 ]; then
    echo "Error: nbstripout encountered issues. Check warnings above."
    # Optionally, you might want to exit here if stripping is critical
    # exit 1
fi

# Stage the stripped changes
echo "Staging stripped notebook changes..."
git add .
if [ $? -ne 0 ]; then
    echo "Error: Could not stage changes. Exiting."
    exit 1
fi

# Commit the stripped changes (temporarily on this branch)
echo "Committing stripped changes..."
git commit -m "Strip notebook outputs for archival on $TEMP_BRANCH_NAME"
if [ $? -ne 0 ]; then
    echo "Error: Could not commit stripped changes. Exiting."
    # This might happen if there were no notebooks or no changes detected by nbstripout
    # In such cases, we can continue as the repo is already "clean" for the zip
fi

# 6. Generate a timestamp for the zip filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ZIP_FULL_FILENAME="${ZIP_FILENAME_PREFIX}_${REPO_NAME}_${TIMESTAMP}.zip"
ZIP_FILE_PATH="$ZIP_DEST_DIR/$ZIP_FULL_FILENAME"

# 7. Zip the entire repository
echo "Creating zip archive: $ZIP_FILE_PATH"
# Zip from the parent directory to include the repo folder itself in the archive
(cd "$PARENT_DIR" && zip -r "$ZIP_FILE_PATH" "$REPO_NAME")
if [ $? -ne 0 ]; then
    echo "Error: Failed to create zip archive. Exiting."
    # Clean up temp branch before exiting on error
    git checkout "$CURRENT_BRANCH" > /dev/null 2>&1
    git branch -D "$TEMP_BRANCH_NAME" > /dev/null 2>&1
    exit 1
fi

echo "Zip archive created successfully."

# 8. Switch back to the original branch and delete the temporary branch
echo "Switching back to original branch: $CURRENT_BRANCH"
git checkout "$CURRENT_BRANCH"
if [ $? -ne 0 ]; then
    echo "Error: Could not switch back to original branch. Please do it manually."
    exit 1
fi

echo "Deleting temporary branch: $TEMP_BRANCH_NAME"
git branch -D "$TEMP_BRANCH_NAME"
if [ $? -ne 0 ]; then
    echo "Warning: Could not delete temporary branch. You may need to delete it manually: 'git branch -D $TEMP_BRANCH_NAME'"
fi

echo "Script finished."
echo "Your zipped repository is located at: $ZIP_FILE_PATH"