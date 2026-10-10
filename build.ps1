param([switch]$AlgoDebug)

$ErrorActionPreference = 'Stop'

# Portable Windows x64 tools. All files stay inside this project.
$cmakeName = 'cmake-4.4.4-windows-x86_64'
$compilerName = 'llvm-mingw-20261006-ucrt-x86_64'
$tools = Join-Path $PSScriptRoot '.tools'
$deps = Join-Path $PSScriptRoot '.deps'
$cmake = Join-Path $tools "$cmakeName/bin/cmake.exe"
$compilerBin = Join-Path $tools "$compilerName/bin"
$ninja = Join-Path $tools 'ninja/ninja.exe'
$opencvPrefix = Join-Path $deps 'opencv'
$opencvSource = Join-Path $deps 'opencv-4.14.0'

function Install-Archive($Url, $Archive, $Destination, $Marker, $Sha256) {
    if (Test-Path -LiteralPath $Marker) { return }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    if (!(Test-Path -LiteralPath $Archive)) {
        Write-Host "Downloading $Url"
        & curl.exe --fail --location --retry 3 --silent --show-error --output $Archive $Url
        if ($LASTEXITCODE -ne 0) {
            Remove-Item -LiteralPath $Archive -Force -ErrorAction SilentlyContinue
            throw "Download failed: $Url"
        }
    }
    if ($Sha256 -and (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash -ne $Sha256) {
        throw "Checksum mismatch: $Archive. Delete this archive and retry."
    }
    Write-Host "Extracting $Archive"
    & tar.exe -xf $Archive -C $Destination
    if ($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath $Marker)) {
        throw "Extraction failed: $Archive"
    }
}

Install-Archive "https://github.com/Kitware/CMake/releases/download/v4.4.4/$cmakeName.zip" `
    "$tools/$cmakeName.zip" $tools $cmake `
    'bace36e94b31c68ab6fa295f26dfa11219e0701cf7c94b0284a7d1cb13dac536'
Install-Archive "https://github.com/mstorsjo/llvm-mingw/releases/download/20261006/$compilerName.zip" `
    "$tools/$compilerName.zip" $tools "$compilerBin/clang++.exe" `
    '317492c456aa27ee607a5919f1d2d38dcdc1112516a24d0bf4b00d078f52d17a'
Install-Archive 'https://github.com/ninja-build/ninja/releases/download/v1.13.2/ninja-win.zip' `
    "$tools/ninja-1.13.2.zip" "$tools/ninja" $ninja `
    '07fc8261b42b20e71d1720b39068c2e14ffcee6396b76fb7a795fb460b78dc65'

$savedPath = $env:PATH
$env:PATH = "$compilerBin;$savedPath"
try {
    $toolchain = @('-G', 'Ninja', "-DCMAKE_MAKE_PROGRAM=$ninja",
        "-DCMAKE_C_COMPILER=$compilerBin/clang.exe", "-DCMAKE_CXX_COMPILER=$compilerBin/clang++.exe",
        '-DCMAKE_BUILD_TYPE=Release')

    if (!(Test-Path -LiteralPath "$opencvPrefix/lib/cmake/opencv4/OpenCVConfig.cmake")) {
        Install-Archive 'https://codeload.github.com/opencv/opencv/tar.gz/refs/tags/4.14.0' `
            "$deps/opencv-4.14.0.tar.gz" $deps "$opencvSource/CMakeLists.txt" `
            'ee8fb9b30eb60850431b4656447080e3737b56e45719c92b67f245950609f86e'
        & $cmake -S $opencvSource -B "$deps/opencv-build" @toolchain `
            -C "$PSScriptRoot/cmake/opencv.cmake" "-DCMAKE_INSTALL_PREFIX=$opencvPrefix"
        if ($LASTEXITCODE -ne 0) { throw 'OpenCV configuration failed' }
        & $cmake --build "$deps/opencv-build" --parallel
        if ($LASTEXITCODE -ne 0) { throw 'OpenCV build failed' }
        & $cmake --install "$deps/opencv-build"
        if ($LASTEXITCODE -ne 0) { throw 'OpenCV installation failed' }
    }

    $debugOption = if ($AlgoDebug) { 'ON' } else { 'OFF' }
    & $cmake -S $PSScriptRoot -B "$PSScriptRoot/build" @toolchain "-DOpenCV_DIR=$opencvPrefix/lib/cmake/opencv4" "-DCOLORIST_DEBUG=$debugOption"
    if ($LASTEXITCODE -ne 0) { throw 'App configuration failed' }
    & $cmake --build "$PSScriptRoot/build" --parallel
    if ($LASTEXITCODE -ne 0) { throw 'App build failed' }
} finally {
    $env:PATH = $savedPath
}
