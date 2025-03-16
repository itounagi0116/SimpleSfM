#!/usr/bin/env python3
"""
SimpleSfM用の拡張メディア対応モジュール
- RAWファイル対応
- WEBPファイル対応
- 動画ファイルからのフレーム抽出
"""

import os
import sys
import numpy as np
import cv2
import rawpy
from pathlib import Path
import logging
import subprocess
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from threading import Lock  # Import Lock for thread-safe operations

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("media_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MediaProcessor:
    """様々なメディアフォーマットを処理するクラス"""

    def __init__(self, output_format='jpg', max_dimension=None, quality=95, temp_dir=None):
        """
        初期化

        Args:
            output_format: 出力画像フォーマット ('jpg', 'png', 'tiff')
            max_dimension: 出力画像の最大サイズ（ピクセル）。Noneの場合は元のサイズを維持
            quality: JPEG出力の品質（1-100）
            temp_dir: 一時ファイルディレクトリ
        """
        self.output_format = output_format.lower()
        self.max_dimension = max_dimension
        self.quality = quality
        self.temp_dir = temp_dir or tempfile.mkdtemp()

        # サポートするファイル拡張子
        self.raw_extensions = ['.arw', '.cr2', '.cr3', '.dng', '.nef', '.orf', '.raf', '.rw2',
                            '.pef', '.srw', '.x3f', '.raw']
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp']
        self.video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm']

        # 一時ディレクトリの作成
        os.makedirs(self.temp_dir, exist_ok=True)

    def __del__(self):
        """デストラクタ - 一時ディレクトリのクリーンアップ"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass

    def process_file(self, file_path, output_dir=None):
        """
        ファイルの種類に応じた処理を実行

        Args:
            file_path: 処理するファイルのパス
            output_dir: 出力ディレクトリ。Noneの場合は一時ディレクトリを使用

        Returns:
            処理された画像ファイルのパスのリスト
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        # 出力ディレクトリの設定
        if output_dir is None:
            output_dir = self.temp_dir
        os.makedirs(output_dir, exist_ok=True)

        try:
            # ファイルタイプに応じた処理
            if extension in self.raw_extensions:
                return [self.process_raw(file_path, output_dir)]
            elif extension == '.webp':
                return [self.process_webp(file_path, output_dir)]
            elif extension in self.video_extensions:
                return self.extract_video_frames(file_path, output_dir)
            elif extension in self.image_extensions:
                # 通常の画像形式はそのままコピー（リサイズ処理のみ）
                return [self.process_regular_image(file_path, output_dir)]
            else:
                logger.warning(f"サポートされていないファイル形式です: {file_path}")
                return []
        except Exception as e:
            logger.error(f"ファイル処理中にエラーが発生しました ({file_path}): {str(e)}")
            return []

    def process_raw(self, file_path, output_dir):
        """
        RAWファイルを処理してJPEG/PNGに変換

        Args:
            file_path: RAWファイルのパス
            output_dir: 出力ディレクトリ

        Returns:
            処理された画像ファイルのパス
        """
        logger.info(f"RAWファイルを処理中: {file_path}")
        output_filename = f"{file_path.stem}.{self.output_format}"
        output_path = os.path.join(output_dir, output_filename)

        try:
            # rawpyを使用してRAWファイルを開く
            with rawpy.imread(str(file_path)) as raw:
                # 画像を処理（自動ホワイトバランス、自動明るさ調整）
                rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=False, output_bps=8)

            # OpenCVフォーマットに変換（RGB→BGR）
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # 必要に応じてリサイズ
            img = self._resize_image(img)

            # 保存
            self._save_image(img, output_path)

            return output_path
        except Exception as e:
            logger.error(f"RAWファイル処理中にエラー: {str(e)}")
            raise

    def process_webp(self, file_path, output_dir):
        """
        WEBPファイルを処理

        Args:
            file_path: WEBPファイルのパス
            output_dir: 出力ディレクトリ

        Returns:
            処理された画像ファイルのパス
        """
        logger.info(f"WEBPファイルを処理中: {file_path}")
        output_filename = f"{file_path.stem}.{self.output_format}"
        output_path = os.path.join(output_dir, output_filename)

        try:
            # PILを使用してWEBPを読み込む
            img = Image.open(file_path)

            # アニメーションWEBPの場合は最初のフレームを使用
            img.seek(0)

            # RGB形式に変換
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # NumPy配列に変換
            img_array = np.array(img)

            # OpenCVフォーマットに変換（RGB→BGR）
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # 必要に応じてリサイズ
            img_cv = self._resize_image(img_cv)

            # 保存
            self._save_image(img_cv, output_path)

            return output_path
        except Exception as e:
            logger.error(f"WEBP処理中にエラー: {str(e)}")
            raise

    def extract_video_frames(self, video_path, output_dir, fps=1, max_frames=100):
        """
        動画ファイルからフレームを抽出

        Args:
            video_path: 動画ファイルのパス
            output_dir: 出力ディレクトリ
            fps: 抽出するフレームレート
            max_frames: 最大抽出フレーム数

        Returns:
            抽出されたフレーム画像のパスのリスト
        """
        logger.info(f"動画ファイルからフレームを抽出中: {video_path}")
        video_path = Path(video_path)

        # 動画用のサブディレクトリを作成
        video_output_dir = os.path.join(output_dir, video_path.stem)
        os.makedirs(video_output_dir, exist_ok=True)

        try:
            # 動画キャプチャの開始
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"動画ファイルを開けませんでした: {video_path}")
                return []

            # 動画情報の取得
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / video_fps

            logger.info(f"動画情報: {total_frames}フレーム, {video_fps}fps, 長さ:{duration:.1f}秒")

            # 抽出間隔の計算
            frame_interval = int(video_fps / fps)
            if frame_interval < 1:
                frame_interval = 1

            # フレーム数の制限
            frames_to_extract = min(max_frames, int(total_frames / frame_interval))
            frame_interval = max(frame_interval, int(total_frames / max_frames))

            logger.info(f"{frames_to_extract}フレームを抽出します（間隔: {frame_interval}フレーム）")

            # フレーム抽出
            extracted_paths = []
            frame_count = 0
            extracted_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # フレーム間隔ごとに画像を保存
                if frame_count % frame_interval == 0:
                    # リサイズ
                    frame = self._resize_image(frame)

                    # 保存
                    output_filename = f"frame_{extracted_count:06d}.{self.output_format}"
                    output_path = os.path.join(video_output_dir, output_filename)
                    self._save_image(frame, output_path)

                    extracted_paths.append(output_path)
                    extracted_count += 1

                    # 最大フレーム数に達したら終了
                    if extracted_count >= frames_to_extract:
                        break

                frame_count += 1

            # リソースの解放
            cap.release()

            logger.info(f"{extracted_count}フレームを正常に抽出しました")
            return extracted_paths

        except Exception as e:
            logger.error(f"動画フレーム抽出中にエラー: {str(e)}")
            raise

    def process_regular_image(self, file_path, output_dir):
        """
        通常の画像ファイル（JPG, PNG, TIFF）を処理

        Args:
            file_path: 画像ファイルのパス
            output_dir: 出力ディレクトリ

        Returns:
            処理された画像ファイルのパス
        """
        output_filename = f"{file_path.stem}.{self.output_format}"
        output_path = os.path.join(output_dir, output_filename)

        try:
            # 画像を読み込む
            img = cv2.imread(str(file_path))
            if img is None:
                raise ValueError(f"画像を読み込めませんでした: {file_path}")

            # リサイズ
            img = self._resize_image(img)

            # 保存
            self._save_image(img, output_path)

            return output_path
        except Exception as e:
            logger.error(f"画像処理中にエラー ({file_path}): {str(e)}")
            raise

    def process_directory(self, input_dir, output_dir, recursive=False, num_threads: int | None = None):
        """
        ディレクトリ内のすべてのメディアファイルを処理

        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
            recursive: サブディレクトリも処理するかどうか
            num_threads: 並列処理用のスレッド数

        Returns:
            処理された画像ファイルのパスのリスト
        """
        input_dir = Path(input_dir)
        os.makedirs(output_dir, exist_ok=True)

        # サポートされているファイル拡張子をすべて結合
        all_extensions = self.raw_extensions + self.image_extensions + self.video_extensions

        # ファイルリストの作成
        files_to_process = []

        if recursive:
            # 再帰的に全ファイルを検索
            for ext in all_extensions:
                files_to_process.extend(input_dir.glob(f"**/*{ext}"))
                files_to_process.extend(input_dir.glob(f"**/*{ext.upper()}"))
        else:
            # 現在のディレクトリのみ
            for ext in all_extensions:
                files_to_process.extend(input_dir.glob(f"*{ext}"))
                files_to_process.extend(input_dir.glob(f"*{ext.upper()}"))

        total_files = len(files_to_process)
        logger.info(f"{total_files}個のメディアファイルが見つかりました")

        if total_files == 0:
            return []

        # 並列処理用のスレッド数
        num_threads = num_threads if num_threads is not None else min(os.cpu_count(), 8)
        logger.info(f"{num_threads}スレッドで処理を実行します")

        # 進捗カウンタとロック
        processed_count = 0
        lock = Lock()

        # 各ファイルを処理する関数
        def process_file_with_progress(file_path):
            nonlocal processed_count
            result = self.process_file(file_path, output_dir)
            with lock:  # Ensure thread-safe increment
                processed_count += 1
                if processed_count % 10 == 0 or processed_count == total_files:
                    logger.info(f"進捗: {processed_count}/{total_files} ファイル処理済み")
            return result

        # 並列処理の実行
        all_processed_files = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(process_file_with_progress, files_to_process))
            for file_list in results:
                if file_list:
                    all_processed_files.extend(file_list)

        logger.info(f"処理完了: {len(all_processed_files)}個の画像ファイルが生成されました")
        return all_processed_files

    def _resize_image(self, img):
        """画像をリサイズする内部メソッド"""
        if self.max_dimension is None:
            return img

        h, w = img.shape[:2]
        if max(h, w) <= self.max_dimension:
            return img

        # アスペクト比を維持してリサイズ
        scale = self.max_dimension / max(h, w)
        new_width = int(w * scale)
        new_height = int(h * scale)

        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def _save_image(self, img, output_path):
        """画像を保存する内部メソッド"""
        if self.output_format == 'jpg' or self.output_format == 'jpeg':
            cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        elif self.output_format == 'png':
            cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        elif self.output_format == 'tiff' or self.output_format == 'tif':
            cv2.imwrite(output_path, img)
        else:
            # デフォルトはJPEG
            if not output_path.endswith('.jpg'):
                output_path += '.jpg'
            cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])

        return output_path


def main():
    """コマンドラインからの実行用メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="RAW/WEBP/動画ファイル処理ツール")
    parser.add_argument("--input", "-i", required=True, help="入力ファイルまたはディレクトリ")
    parser.add_argument("--output", "-o", required=True, help="出力ディレクトリ")
    parser.add_argument("--format", "-f", choices=["jpg", "png", "tiff"], default="jpg",
                        help="出力画像フォーマット")
    parser.add_argument("--max-size", "-s", type=int, default=3000,
                        help="出力画像の最大サイズ（ピクセル）")
    parser.add_argument("--quality", "-q", type=int, default=95,
                        help="JPEG出力の品質（1-100）")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="サブディレクトリも処理")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="動画からのフレーム抽出レート")
    parser.add_argument("--max-frames", type=int, default=100,
                        help="動画からの最大フレーム抽出数")
    parser.add_argument("--threads", "-t", type=int, default=None,
                        help="並列処理用のスレッド数")

    args = parser.parse_args()

    # 処理クラスの初期化
    processor = MediaProcessor(
        output_format=args.format,
        max_dimension=args.max_size,
        quality=args.quality
    )

    # 入力がファイルかディレクトリかを判断
    input_path = Path(args.input)

    if input_path.is_file():
        # 単一ファイルの処理
        processor.process_file(input_path, args.output)
    elif input_path.is_dir():
        # ディレクトリの処理
        processor.process_directory(
            args.input,
            args.output,
            recursive=args.recursive,
            num_threads=args.threads
        )
    else:
        logger.error(f"入力パスが見つかりません: {args.input}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
