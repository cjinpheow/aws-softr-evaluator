## Accessing CodeCommit with AWS IAM Identity Center (SSO)

### Signing in

Log in to **AWS Identity Center (SSO)** with the following command:

```
aws sso login --profile aws-profile-name
```

### Installing git-remote-codecommit

Install **git-remote-codecommit** with the following command:

```
pip install git-remote-codecommit
```

### Cloning a repository

Clone your **CodeCommit** repository with the following command:

```
git clone codecommit://aws-profile-name@repository-name local-folder
```

That's it!
