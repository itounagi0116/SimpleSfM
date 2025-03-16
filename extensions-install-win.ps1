# SimpleSfM Windows インストールスクリプト (PowerShell)

Write-Host "SimpleSfM Windows インストールスクリプト" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green

# Python確認
$pythonCommand = $null
try {
    python --version | Out-Null
    $pythonCommand = "python"
} catch {
    try {
        py --version | Out-Null
        $pythonCommand = "py"
    } catch {
        Write-Host "Pythonが見つかりません。Python 3.7以上をインストールしてください。" -ForegroundColor Red
        Write-Host "https://www.python.org/downloads/windows/ からダウンロードできます。" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "Pythonを確認しました: $pythonCommand" -ForegroundColor Green

# pipの確認
try {
    & $pythonCommand -m pip --version | Out-Null
} catch {
    Write-Host "pipが見つかりません。" -ForegroundColor Red
    exit 1
}

# 仮想環境の作成（オプション）
$createVenv = Read-Host "Python仮想環境を作成しますか？ [y/N]"
if ($createVenv -eq "y" -or $createVenv -eq "Y") {
    Write-Host "仮想環境を作成中..." -ForegroundColor Yellow
    & $pythonCommand -m pip install virtualenv
    & $pythonCommand -m virtualenv venv
    
    # 仮想環境のアクティベート
    if (Test-Path ".\venv\Scripts\activate.ps1") {
        . .\venv\Scripts\activate.ps1
        Write-Host "仮想環境をアクティベートしました。" -ForegroundColor Green
    } else {
        Write-Host "仮想環境のアクティベートに失敗しました。" -ForegroundColor Red
        exit 1
    }
}

# 基本依存関係のインストール
Write-Host "基本依存関係をインストール中..." -ForegroundColor Yellow
& $pythonCommand -m pip install numpy opencv-python open3d pyqt5 pycolmap

# 拡張機能のインストール確認
Write-Host "`n拡張機能のインストール" -ForegroundColor Green
Write-Host "======================" -ForegroundColor Green

# RAW/WEBP/動画対応拡張機能のインストール
$installRawWebpVideo = Read-Host "RAW/WEBP/動画対応拡張機能をインストールしますか？ [y/N]"
if ($installRawWebpVideo -eq "y" -or $installRawWebpVideo -eq "Y") {
    Write-Host "RAW/WEBP/動画対応拡張機能をインストール中..." -ForegroundColor Yellow
    & $pythonCommand -m pip install rawpy pillow psutil
    
    # ExifTool確認・インストール
    try {
        exiftool -ver | Out-Null
        Write-Host "ExifToolが見つかりました。" -ForegroundColor Green
    } catch {
        Write-Host "ExifToolが見つかりません。手動でインストールしてください。" -ForegroundColor Yellow
        Write-Host "https://exiftool.org/ からWindows実行ファイルをダウンロードし、実行パスに配置してください。" -ForegroundColor Yellow
    }
    
    # FFmpeg確認・インストール
    try {
        ffmpeg -version | Out-Null
        Write-Host "FFmpegが見つかりました。" -ForegroundColor Green
    } catch {
        Write-Host "FFmpegが見つかりません。手動でインストールしてください。" -ForegroundColor Yellow
        Write-Host "https://ffmpeg.org/download.html からWindows版をダウンロードし、実行パスに配置してください。" -ForegroundColor Yellow
    }
    
    Write-Host "RAW/WEBP/動画対応拡張機能のインストールが完了しました。" -ForegroundColor Green
}

# 空間メディア対応拡張機能のインストール
$installSpatialMedia = Read-Host "空間メディア対応拡張機能（Insta360, XREAL等）をインストールしますか？ [y/N]"
if ($installSpatialMedia -eq "y" -or $installSpatialMedia -eq "Y") {
    Write-Host "空間メディア対応拡張機能をインストール中..." -ForegroundColor Yellow
    & $pythonCommand -m pip install exifread pyvista pyexiftool
    
    # ExifTool確認（上記と同じ）
    if (-not (Get-Command exiftool -ErrorAction SilentlyContinue)) {
        Write-Host "ExifToolが見つかりません。手動でインストールしてください。" -ForegroundColor Yellow
        Write-Host "https://exiftool.org/ からWindows実行ファイルをダウンロードし、実行パスに配置してください。" -ForegroundColor Yellow
    }
    
    # FFmpeg確認（上記と同じ）
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
        Write-Host "FFmpegが見つかりません。手動でインストールしてください。" -ForegroundColor Yellow
        Write-Host "https://ffmpeg.org/download.html からWindows版をダウンロードし、実行パスに配置してください。" -ForegroundColor Yellow
    }
    
    Write-Host "空間メディア対応拡張機能のインストールが完了しました。" -ForegroundColor Green
}

Write-Host "`nSimpleSfMのインストールが完了しました！" -ForegroundColor Green
Write-Host "以下のコマンドで実行できます：" -ForegroundColor Yellow
Write-Host "python simpleSfM.py" -ForegroundColor White
Write-Host "`n仮想環境を使用している場合は、使用前に必ず以下のコマンドで仮想環境をアクティベートしてください：" -ForegroundColor Yellow
Write-Host ".\venv\Scripts\activate.ps1" -ForegroundColor White
