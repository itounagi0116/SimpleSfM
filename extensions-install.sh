#!/bin/bash
# SimpleSfM - 総合インストールスクリプト

# 色の定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}SimpleSfM インストールスクリプト${NC}"
echo "=============================="
echo

# コマンドが利用可能かチェック
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}$1 が見つかりません。インストールが必要です。${NC}"
        return 1
    else
        echo -e "${GREEN}$1 が見つかりました。${NC}"
        return 0
    fi
}

# OS判定
if [ -f /etc/os-release ]; then
    # freedesktop.org and systemd
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
elif type lsb_release >/dev/null 2>&1; then
    # linuxbase.org
    OS=$(lsb_release -si)
    VER=$(lsb_release -sr)
elif [ -f /etc/lsb-release ]; then
    # For some versions of Debian/Ubuntu without lsb_release command
    . /etc/lsb-release
    OS=$DISTRIB_ID
    VER=$DISTRIB_RELEASE
elif [ -f /etc/debian_version ]; then
    # Older Debian/Ubuntu/etc.
    OS=Debian
    VER=$(cat /etc/debian_version)
elif [ -f /etc/SuSe-release ]; then
    # Older SuSE/etc.
    OS=SuSE
elif [ -f /etc/redhat-release ]; then
    # Older Red Hat, CentOS, etc.
    OS=RedHat
else
    # Fall back to uname, e.g. "Linux <version>", also works for BSD, etc.
    OS=$(uname -s)
    VER=$(uname -r)
fi

echo -e "${YELLOW}検出されたOS: $OS $VER${NC}"
echo

# Python確認
echo "Pythonを確認中..."
if check_command python3; then
    PYTHON=python3
elif check_command python; then
    PYTHON=python
else
    echo -e "${RED}Pythonが見つかりません。まずPython 3.7以上をインストールしてください。${NC}"
    exit 1
fi

# pipの確認
echo "Pipを確認中..."
if ! check_command pip3 && ! check_command pip; then
    echo -e "${RED}pipが見つかりません。インストールしてください。${NC}"
    exit 1
fi

if check_command pip3; then
    PIP=pip3
else
    PIP=pip
fi

# 仮想環境の作成（オプション）
read -p "Python仮想環境を作成しますか？ [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "仮想環境を作成中..."
    if ! check_command virtualenv; then
        $PIP install virtualenv
    fi
    virtualenv venv
    
    # 仮想環境のアクティベート
    if [ -f venv/bin/activate ]; then
        source venv/bin/activate
        echo -e "${GREEN}仮想環境をアクティベートしました。${NC}"
    else
        echo -e "${RED}仮想環境のアクティベートに失敗しました。${NC}"
        exit 1
    fi
fi

# 基本依存関係のインストール
echo "基本依存関係をインストール中..."
$PIP install numpy opencv-python open3d pyqt5 pycolmap

# システムに応じた依存関係インストール
if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
    echo "Ubuntu/Debian向けシステムライブラリをインストール中..."
    sudo apt-get update
    sudo apt-get install -y build-essential libgl1-mesa-glx libgomp1
elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Fedora"* ]] || [[ "$OS" == *"Red Hat"* ]]; then
    echo "CentOS/Fedora/Red Hat向けシステムライブラリをインストール中..."
    sudo yum install -y gcc-c++ mesa-libGL mesa-libGLU libgomp
elif [[ "$OS" == *"Arch"* ]]; then
    echo "Arch Linux向けシステムライブラリをインストール中..."
    sudo pacman -S --noconfirm gcc mesa libgomp
elif [[ "$OS" == *"Mac"* ]] || [[ "$OS" == *"Darwin"* ]]; then
    echo "macOS向けライブラリをインストール中..."
    if ! check_command brew; then
        echo -e "${RED}Homebrewが見つかりません。Homebrewをインストールしてから再実行してください。${NC}"
        echo "Homebrewのインストールコマンド: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    brew install libomp
fi

# 拡張機能のインストール確認
echo
echo "拡張機能のインストール"
echo "======================"

# RAW/WEBP/動画対応拡張機能のインストール
read -p "RAW/WEBP/動画対応拡張機能をインストールしますか？ [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "RAW/WEBP/動画対応拡張機能をインストール中..."
    $PIP install rawpy pillow psutil
    
    # ExifToolのインストール（プラットフォームに応じた方法）
    if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
        sudo apt-get install -y libimage-exiftool-perl
    elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Fedora"* ]] || [[ "$OS" == *"Red Hat"* ]]; then
        sudo yum install -y perl-Image-ExifTool
    elif [[ "$OS" == *"Arch"* ]]; then
        sudo pacman -S --noconfirm perl-image-exiftool
    elif [[ "$OS" == *"Mac"* ]] || [[ "$OS" == *"Darwin"* ]]; then
        brew install exiftool
    else
        echo -e "${YELLOW}ExifToolの自動インストールはこのOSではサポートされていません。手動でインストールしてください。${NC}"
        echo "https://exiftool.org/ からダウンロードしてインストールしてください。"
    fi
    
    # FFmpegのインストール
    if ! check_command ffmpeg; then
        if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
            sudo apt-get install -y ffmpeg
        elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Fedora"* ]] || [[ "$OS" == *"Red Hat"* ]]; then
            sudo yum install -y ffmpeg
        elif [[ "$OS" == *"Arch"* ]]; then
            sudo pacman -S --noconfirm ffmpeg
        elif [[ "$OS" == *"Mac"* ]] || [[ "$OS" == *"Darwin"* ]]; then
            brew install ffmpeg
        else
            echo -e "${YELLOW}FFmpegの自動インストールはこのOSではサポートされていません。手動でインストールしてください。${NC}"
            echo "https://ffmpeg.org/download.html からダウンロードしてインストールしてください。"
        fi
    fi
    
    echo -e "${GREEN}RAW/WEBP/動画対応拡張機能のインストールが完了しました。${NC}"
fi

# 空間メディア対応拡張機能のインストール
read -p "空間メディア対応拡張機能（Insta360, XREAL等）をインストールしますか？ [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "空間メディア対応拡張機能をインストール中..."
    $PIP install exifread pyvista pyexiftool
    
    # ExifToolのインストール（上記と同様）
    if ! check_command exiftool; then
        if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
            sudo apt-get install -y libimage-exiftool-perl
        elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Fedora"* ]] || [[ "$OS" == *"Red Hat"* ]]; then
            sudo yum install -y perl-Image-ExifTool
        elif [[ "$OS" == *"Arch"* ]]; then
            sudo pacman -S --noconfirm perl-image-exiftool
        elif [[ "$OS" == *"Mac"* ]] || [[ "$OS" == *"Darwin"* ]]; then
            brew install exiftool
        else
            echo -e "${YELLOW}ExifToolの自動インストールはこのOSではサポートされていません。手動でインストールしてください。${NC}"
        fi
    fi
    
    # FFmpegのインストール（上記と同様）
    if ! check_command ffmpeg; then
        if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
            sudo apt-get install -y ffmpeg
        elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Fedora"* ]] || [[ "$OS" == *"Red Hat"* ]]; then
            sudo yum install -y ffmpeg
        elif [[ "$OS" == *"Arch"* ]]; then
            sudo pacman -S --noconfirm ffmpeg
        elif [[ "$OS" == *"Mac"* ]] || [[ "$OS" == *"Darwin"* ]]; then
            brew install ffmpeg
        else
            echo -e "${YELLOW}FFmpegの自動インストールはこのOSではサポートされていません。手動でインストールしてください。${NC}"
        fi
    fi
    
    echo -e "${GREEN}空間メディア対応拡張機能のインストールが完了しました。${NC}"
fi

echo
echo -e "${GREEN}SimpleSfMのインストールが完了しました！${NC}"
echo "以下のコマンドで実行できます："
echo "python3 simpleSfM.py"
echo
echo "仮想環境を使用している場合は、使用前に必ず以下のコマンドで仮想環境をアクティベートしてください："
echo "source venv/bin/activate"
