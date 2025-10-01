# return the path to $project_root
param(
    [Parameter(Mandatory=$false)][string]$project_root="AI_from_scratch"
)

$list_paths=(& Get-Location).Path.Split('\')

# find the index in list_paths at which we get the project root
# and determine its absolute path
$index = 0
$root_path = ""
while (!($list_paths[$index] -eq $project_root)) {
    $root_path = $root_path, $list_paths[$index] -join "\"
    $index = $index + 1
}

$root_path = $root_path, $project_root -join "\"
return $root_path.TrimStart("\")
