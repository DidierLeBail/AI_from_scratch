param(
    [Parameter(Mandatory=$false)][string]$commit_msg="minor"
)

$remote="origin"
$current_branch=& git rev-parse --abbrev-ref HEAD

# push local changes upstream
& git add ../*
& git commit -m $commit_msg
& git push $remote $current_branch
