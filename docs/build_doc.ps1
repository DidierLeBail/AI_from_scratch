# build the docs as html: then drag and drop the build/html/index.html file in a browser

# remove the folder source/_autosummary
$path = Join-Path -Path $PWD -ChildPath "source/_autosummary"
if (Test-Path -Path $path) {
    & Remove-Item -Path $path -Recurse -Force
}

& .\make clean
& .\make html
