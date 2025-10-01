# create a new branch both locally and upstream, and then switch back to the current branch
param(
    [Parameter(Mandatory=$true)][string]$new_branch
)
$current_branch=& git rev-parse --abbrev-ref HEAD
$commit_msg="initialized a new branch"

& git checkout -b $new_branch

.\fast_push.ps1 $commit_msg

& git checkout $current_branch
