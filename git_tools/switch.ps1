param(
    [Parameter(Mandatory=$true)][string]$new_branch
)

# first push eventual changes
.\fast_push.ps1

# get the path to the root of the project
$root_path = .\get_root_path.ps1

# remove undesired untracked files:
#__pycache__
& Get-ChildItem -Path $root_path -Include __pycache__ -Recurse | Remove-Item -Recurse -Force

# switch branch
& git checkout $new_branch
