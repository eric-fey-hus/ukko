# Howto: Transfer a git repo with all commits to Acamedic

Use these instructions for downloading and transferring a Git repository to a system without internet access while preserving all commits and history. 

__Best practise__ is to use a new mirror clone for this.

The main advantage of using --mirror is when you want to create an exact mirror of a remote repository, typically for backup or mirroring purposes. 

The --mirror option creates a bare clone that is an exact mirror of the source repository, including:

- All remote branches
- All refs (including remote-tracking branches, notes, and tags)
- All Git configuration settings

The key differences from a regular clone are:

- It creates a bare repository (no working directory)
- It mirrors all refs exactly as they are in the remote
- All local references will be overwritten each time you fetch, making it always exactly match the original repository

__Alternative__ when you already have a working clone, is to create the bundle from there.  This might be more appropriate if you want to preserves your existing work while including all the history and branches.

---

## Best practise approach: Clone --mirror

Steps:

1. First, on a system WITH internet access:

   a. Create a complete mirror clone of the repository using:
   ```bash
   git clone --mirror https://github.com/user/repo.git
   ```
   This will create a bare repository with all commits, branches, and history.

   b. If the repository has Git LFS (Large File Storage) objects, you'll also need to fetch those:
   ```bash
   cd repo.git
   git lfs fetch --all
   ```

   c. Create a bundle of the repository:
   ```bash
   git bundle create repo.bundle --all
   ```

2. Transfer the files to the offline system:
   - Copy the `repo.bundle` file
   - If there are LFS objects, copy the `.git/lfs` directory as well

3. On the offline system:

   a. Create a new directory for your repository and navigate to it:
   ```bash
   mkdir repo
   cd repo
   ```

   b. Clone from the bundle:
   ```bash
   git clone repo.bundle .
   ```

   c. If you have LFS objects, copy the LFS files to the appropriate location in the `.git/lfs` directory

4. Set up the remote (optional):
   ```bash
   git remote add origin /path/to/local/bare/repository
   ```

Important notes:
- The bundle file contains all repository data including the full commit history
- Make sure you have enough storage space, as you'll be transferring the complete repository history
- If the repository uses Git LFS, you'll need to handle those files separately as mentioned above
- You might need to reconfigure any repository-specific settings after the transfer

This method ensures you have a complete copy of the repository with all commits, branches, and history intact, which can be used on a system without internet access.

For more detailed information about working with remote repositories, you can refer to [About remote repositories](https://docs.github.com/en/get-started/git-basics/about-remote-repositories).

---

## Alternative approach: Create bundle from working clone

For your existing cloned repository, here's how to create a bundle for offline transfer:

1. First, navigate to your existing clone:
    ```bash
    cd your-repository-directory
    ```

2. Create a bundle of all branches and history:
    ```bash
    git bundle create ../repo.bundle --all
    ```

    If your repository has Git LFS objects, you'll need to handle those separately:
    ```bash
    git lfs fetch --all
    ```

3. Then you can copy both:
   - The `repo.bundle` file
   - The `.git/lfs` directory (if you have LFS objects)

To use the bundle on the offline system:

1. Create a new directory and navigate to it:
    ```bash
    mkdir new-repo
    cd new-repo
    ```

2. Clone from the bundle:
    ```bash
    git clone ../repo.bundle .
    ```

For more detailed information, you can refer to [Duplicating a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/duplicating-a-repository) and [Backing up a repository](https://docs.github.com/en/repositories/archiving-a-github-repository/backing-up-a-repository).