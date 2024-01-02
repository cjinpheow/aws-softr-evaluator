## Access AWS CodeCommit repository using AWS IAM Identity Center

### Signing in

Log in to **AWS IAM Identity Center** with the following command:

```
aws sso login --profile aws-profile-name
```

### Installing `git-remote-codecommit`

Install **`git-remote-codecommit`** with the following command:

```
pip install git-remote-codecommit
```

### Cloning a repository

Clone your **AWS CodeCommit** repository with the following command:

```
git clone codecommit://aws-profile-name@repository-name [local-folder]
```

That's it!
