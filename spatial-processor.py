#!/usr/bin/env python3
"""
SimpleSfM用の空間撮影フォーマット対応処理モジュール
- Insta360フォーマット (.insp, .insv)
- XREAL / 空間撮影データ対応
- 空間メタデータ処理
"""

import os
import sys
import json
import numpy as np
import cv2
import tempfile
import shutil
import logging
import subprocess
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
from xml.etree import ElementTree

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("spatial_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SpatialMediaProcessor:
    """空間撮影メディア処理クラス"""
    
    def __init__(self, output_format='jpg', max_dimension=None, quality=95, temp_dir=None,
                equirect_to_perspective=True, extract_stereo=True):
        """
        初期化
        
        Args:
            output_format: 出力画像フォーマット ('jpg', 'png', 'tiff')
            max_dimension: 出力画像の最大サイズ（ピクセル）。Noneの場合は元のサイズを維持
            quality: JPEG出力の品質（1-100）
            temp_dir: 一時ファイルディレクトリ
            equirect_to_perspective: 全天球画像から複数の平面透視投影画像を生成するか
            extract_stereo: ステレオ画像から左右の画像を別々に抽出するか
        """
        self.output_format = output_format.lower()
        self.max_dimension = max_dimension
        self.quality = quality
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.equirect_to_perspective = equirect_to_perspective
        self.extract_stereo = extract_stereo
        
        # サポートするファイル拡張子
        self.insta360_extensions = ['.insp', '.insv']
        self.standard_360_extensions = ['.jpg', '.png', '.mp4']  # 360メタデータを持つ可能性がある標準フォーマット
        self.spatial_extensions = ['.glb', '.gltf']  # 3Dモデル・空間データ
        
        # 一時ディレクトリの作成
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 外部ツールのチェック
        self.check_external_tools()
        
    def __del__(self):
        """デストラクタ - 一時ディレクトリのクリーンアップ"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def check_external_tools(self):
        """必要な外部ツールをチェック"""
        self.exiftool_available = self._check_command('exiftool -ver')
        self.ffmpeg_available = self._check_command('ffmpeg -version')
        
        if not self.exiftool_available:
            logger.warning("ExifTool が見つかりません。空間メタデータの一部機能が制限されます。")
        
        if not self.ffmpeg_available:
            logger.warning("FFmpeg が見つかりません。動画処理機能が制限されます。")
    
    def _check_command(self, command):
        """コマンドが使用可能かチェック"""
        try:
            subprocess.check_output(command.split(), stderr=subprocess.STDOUT)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def process_file(self, file_path, output_dir=None):
        """
        空間撮影ファイルの処理
        
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
            if extension in self.insta360_extensions:
                return self.process_insta360_file(file_path, output_dir)
            elif extension in self.standard_360_extensions:
                # 標準ファイルだが360メタデータがあるか確認
                if self.is_360_media(file_path):
                    return self.process_standard_360_file(file_path, output_dir)
                else:
                    logger.info(f"360メタデータが見つかりません: {file_path}. 通常のメディアとして処理します。")
                    # 通常のメディアとして処理（別モジュールに委譲）
                    return []
            elif extension in self.spatial_extensions:
                logger.info(f"3Dモデル・空間データファイルを検出: {file_path}")
                # 3Dモデルの処理はここでは行わず、情報のみ表示
                return []
            else:
                logger.warning(f"サポートされていない空間ファイル形式です: {file_path}")
                return []
        except Exception as e:
            logger.error(f"空間ファイル処理中にエラーが発生しました ({file_path}): {str(e)}")
            return []
    
    def is_360_media(self, file_path):
        """
        ファイルが360度メディアかどうかを確認
        
        Args:
            file_path: チェックするファイルのパス
            
        Returns:
            360度メディアの場合はTrue、それ以外はFalse
        """
        if not self.exiftool_available:
            logger.warning("ExifToolが利用できないため、360メタデータの確認はスキップします")
            return False
            
        try:
            # ExifToolを使用してメタデータを確認
            cmd = ['exiftool', '-json', '-ProjectionType', '-UsePanoramaViewer', '-GSpherical:Spherical',
                  '-XMP:ProjectionType', '-XMP-GPano:FullPanoWidthPixels', str(file_path)]
            result = subprocess.check_output(cmd, universal_newlines=True)
            
            metadata = json.loads(result)
            if not metadata:
                return False
                
            metadata = metadata[0]  # 最初の要素を取得
            
            # 360メタデータの確認
            is_360 = False
            
            # 様々なメタデータフィールドをチェック
            if 'ProjectionType' in metadata and metadata['ProjectionType'] == 'equirectangular':
                is_360 = True
            elif 'UsePanoramaViewer' in metadata and metadata['UsePanoramaViewer'] == 'True':
                is_360 = True
            elif 'GSpherical:Spherical' in metadata and metadata['GSpherical:Spherical'] == 'True':
                is_360 = True
            elif 'XMP:ProjectionType' in metadata and metadata['XMP:ProjectionType'] == 'equirectangular':
                is_360 = True
            elif 'XMP-GPano:FullPanoWidthPixels' in metadata and metadata['XMP-GPano:FullPanoWidthPixels']:
                is_360 = True
                
            return is_360
            
        except Exception as e:
            logger.warning(f"メタデータ確認中にエラー: {str(e)}")
            return False
    
    def is_stereo_media(self, file_path):
        """
        ファイルがステレオ(3D)メディアかどうかを確認
        
        Args:
            file_path: チェックするファイルのパス
            
        Returns:
            ステレオメディアの場合はTrue、それ以外はFalse
        """
        if not self.exiftool_available:
            logger.warning("ExifToolが利用できないため、ステレオメタデータの確認はスキップします")
            return False
            
        try:
            # ExifToolを使用してメタデータを確認
            cmd = ['exiftool', '-json', '-StereoMode', '-XMP-GImage:Stereoscopic', str(file_path)]
            result = subprocess.check_output(cmd, universal_newlines=True)
            
            metadata = json.loads(result)
            if not metadata:
                return False
                
            metadata = metadata[0]  # 最初の要素を取得
            
            # ステレオメタデータの確認
            is_stereo = False
            
            if 'StereoMode' in metadata and metadata['StereoMode']:
                is_stereo = True
            elif 'XMP-GImage:Stereoscopic' in metadata and metadata['XMP-GImage:Stereoscopic'] == 'True':
                is_stereo = True
                
            # ファイル名でも判定（SBS/TB等のキーワード）
            filename = file_path.name.lower()
            if re.search(r'(sbs|side.?by.?side|lr|left.?right|3d.?180)', filename):
                is_stereo = True
            
            return is_stereo
            
        except Exception as e:
            logger.warning(f"ステレオメタデータ確認中にエラー: {str(e)}")
            return False
    
    def process_insta360_file(self, file_path, output_dir):
        """
        Insta360ファイル (.insp, .insv) の処理
        
        Args:
            file_path: Insta360ファイルのパス
            output_dir: 出力ディレクトリ
            
        Returns:
            処理された画像ファイルのパスのリスト
        """
        logger.info(f"Insta360ファイルを処理中: {file_path}")
        extension = file_path.suffix.lower()
        
        if extension == '.insp':
            # .inspファイル（Insta360の静止画）
            return self.process_insp_file(file_path, output_dir)
        elif extension == '.insv':
            # .insvファイル（Insta360の動画）
            return self.process_insv_file(file_path, output_dir)
        else:
            logger.warning(f"未対応のInsta360形式: {extension}")
            return []
    
    def process_insp_file(self, file_path, output_dir):
        """
        Insta360の静止画ファイル (.insp) の処理
        
        Args:
            file_path: .inspファイルのパス
            output_dir: 出力ディレクトリ
            
        Returns:
            処理された画像ファイルのパスのリスト
        """
        try:
            # .inspファイルはJPEGベースだが専用のヘッダを持つ
            # 一度一時ファイルとしてJPEGに変換
            temp_jpg = os.path.join(self.temp_dir, f"{file_path.stem}.jpg")
            
            # ヘッダーをスキップしてJPEGデータを抽出
            with open(file_path, 'rb') as insp_file, open(temp_jpg, 'wb') as jpg_file:
                # 最初の数バイトはヘッダ情報なのでスキップ
                # 実際にはもっと複雑だが、ここでは簡略化
                insp_file.seek(512)  # 仮のオフセット値
                jpg_file.write(insp_file.read())
            
            # OpenCVで読み込み
            img = cv2.imread(temp_jpg)
            if img is None:
                # ヘッダが異なる場合は、いくつかの一般的なオフセットを試す
                offsets = [0, 128, 256, 384, 512, 1024]
                for offset in offsets:
                    with open(file_path, 'rb') as insp_file, open(temp_jpg, 'wb') as jpg_file:
                        insp_file.seek(offset)
                        jpg_file.write(insp_file.read())
                    img = cv2.imread(temp_jpg)
                    if img is not None:
                        logger.info(f"オフセット {offset} でJPEGデータを見つけました")
                        break
            
            if img is None:
                # まだ読み込めない場合はFFmpegを試す
                if self.ffmpeg_available:
                    cmd = ['ffmpeg', '-i', str(file_path), '-y', temp_jpg]
                    subprocess.run(cmd, stderr=subprocess.PIPE)
                    img = cv2.imread(temp_jpg)
            
            if img is None:
                logger.error(f".inspファイルの変換に失敗: {file_path}")
                return []
            
            # 抽出成功
            # この時点で全天球（360度）画像である可能性が高い
            # メタデータで確認
            is_360 = True  # .inspは基本的に360度画像と仮定
            is_stereo = self.is_stereo_media(temp_jpg)
            
            return self.process_360_image(img, file_path.stem, output_dir, is_360, is_stereo)
            
        except Exception as e:
            logger.error(f".inspファイル処理中にエラー: {str(e)}")
            return []
    
    def process_insv_file(self, file_path, output_dir):
        """
        Insta360の動画ファイル (.insv) の処理
        
        Args:
            file_path: .insvファイルのパス
            output_dir: 出力ディレクトリ
            
        Returns:
            処理された画像ファイルのパスのリスト
        """
        if not self.ffmpeg_available:
            logger.error("FFmpegが利用できないため、.insvファイルを処理できません")
            return []
        
        try:
            # .insvファイルをMP4に変換
            temp_mp4 = os.path.join(self.temp_dir, f"{file_path.stem}.mp4")
            
            cmd = ['ffmpeg', '-i', str(file_path), '-c', 'copy', '-y', temp_mp4]
            subprocess.run(cmd, stderr=subprocess.PIPE)
            
            if not os.path.exists(temp_mp4) or os.path.getsize(temp_mp4) < 1000:
                logger.error(f".insvファイルのMP4変換に失敗: {file_path}")
                return []
            
            # 変換したMP4から一定間隔でフレームを抽出
            video_output_dir = os.path.join(output_dir, file_path.stem)
            os.makedirs(video_output_dir, exist_ok=True)
            
            # 動画が360度かどうかを確認
            is_360 = True  # .insvは基本的に360度動画と仮定
            is_stereo = self.is_stereo_media(temp_mp4)
            
            # フレーム抽出
            fps = 1  # 1秒に1フレーム
            max_frames = 100  # 最大フレーム数
            
            # 動画情報の取得
            cap = cv2.VideoCapture(temp_mp4)
            if not cap.isOpened():
                logger.error(f"変換後のMP4ファイルを開けませんでした: {temp_mp4}")
                return []
            
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
                    # 360度画像として処理
                    frame_output_list = self.process_360_image(
                        frame, 
                        f"{file_path.stem}_frame_{extracted_count:06d}", 
                        video_output_dir,
                        is_360,
                        is_stereo
                    )
                    
                    extracted_paths.extend(frame_output_list)
                    extracted_count += 1
                    
                    # 最大フレーム数に達したら終了
                    if extracted_count >= frames_to_extract:
                        break
                
                frame_count += 1
            
            # リソースの解放
            cap.release()
            
            logger.info(f"{extracted_count}フレームを正常に抽出しました")
            
            # 一時ファイルの削除
            try:
                os.remove(temp_mp4)
            except:
                pass
            
            return extracted_paths
            
        except Exception as e:
            logger.error(f".insvファイル処理中にエラー: {str(e)}")
            return []
    
    def process_standard_360_file(self, file_path, output_dir):
        """
        標準形式の360度メディアファイルの処理
        
        Args:
            file_path: 処理するファイルのパス
            output_dir: 出力ディレクトリ
            
        Returns:
            処理された画像ファイルのパスのリスト
        """
        extension = file_path.suffix.lower()
        
        try:
            if extension in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.webp']:
                # 静止画の処理
                img = cv2.imread(str(file_path))
                if img is None:
                    logger.error(f"画像を読み込めませんでした: {file_path}")
                    return []
                
                is_stereo = self.is_stereo_media(file_path)
                return self.process_360_image(img, file_path.stem, output_dir, True, is_stereo)
                
            elif extension in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
                # 動画の処理
                if not self.ffmpeg_available:
                    logger.error("FFmpegが利用できないため、動画を処理できません")
                    return []
                
                # 一定間隔でフレームを抽出
                video_output_dir = os.path.join(output_dir, file_path.stem)
                os.makedirs(video_output_dir, exist_ok=True)
                
                is_stereo = self.is_stereo_media(file_path)
                
                # フレーム抽出
                fps = 1  # 1秒に1フレーム
                max_frames = 100  # 最大フレーム数
                
                cap = cv2.VideoCapture(str(file_path))
                if not cap.isOpened():
                    logger.error(f"動画ファイルを開けませんでした: {file_path}")
                    return []
                
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_interval = int(video_fps / fps)
                if frame_interval < 1:
                    frame_interval = 1
                
                frames_to_extract = min(max_frames, int(total_frames / frame_interval))
                
                extracted_paths = []
                frame_count = 0
                extracted_count = 0
                
                while extracted_count < frames_to_extract:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % frame_interval == 0:
                        # 360度画像として処理
                        frame_output_list = self.process_360_image(
                            frame,
                            f"{file_path.stem}_frame_{extracted_count:06d}",
                            video_output_dir,
                            True,
                            is_stereo
                        )
                        
                        extracted_paths.extend(frame_output_list)
                        extracted_count += 1
                    
                    frame_count += 1
                
                cap.release()
                return extracted_paths
                
            else:
                logger.warning(f"未対応の360メディア形式: {extension}")
                return []
                
        except Exception as e:
            logger.error(f"360メディア処理中にエラー: {str(e)}")
            return []
    
    def process_360_image(self, img, base_name, output_dir, is_360=True, is_stereo=False):
        """
        360度画像を処理
        
        Args:
            img: OpenCV画像データ
            base_name: 出力ファイルのベース名
            output_dir: 出力ディレクトリ
            is_360: 360度画像かどうか
            is_stereo: ステレオ画像かどうか
            
        Returns:
            処理された画像ファイルのパスのリスト
        """
        output_paths = []
        
        try:
            # ステレオ画像の場合、左右に分割
            if is_stereo and self.extract_stereo:
                h, w = img.shape[:2]
                
                # 左右または上下に分割されているか判断（幅が高さの2倍以上なら左右、それ以外なら上下と仮定）
                if w >= h * 2:  # Side-by-Side
                    # 画像を左右に分割
                    img_left = img[:, :w//2]
                    img_right = img[:, w//2:]
                    
                    # 左目用画像の処理
                    left_paths = self._process_single_view(img_left, f"{base_name}_left", output_dir, is_360)
                    output_paths.extend(left_paths)
                    
                    # 右目用画像の処理
                    right_paths = self._process_single_view(img_right, f"{base_name}_right", output_dir, is_360)
                    output_paths.extend(right_paths)
                    
                else:  # Top-Bottom
                    # 画像を上下に分割
                    img_top = img[:h//2, :]
                    img_bottom = img[h//2:, :]
                    
                    # 上（左目用）画像の処理
                    top_paths = self._process_single_view(img_top, f"{base_name}_top", output_dir, is_360)
                    output_paths.extend(top_paths)
                    
                    # 下（右目用）画像の処理
                    bottom_paths = self._process_single_view(img_bottom, f"{base_name}_bottom", output_dir, is_360)
                    output_paths.extend(bottom_paths)
            else:
                # 通常の360度画像または非ステレオ画像
                output_paths = self._process_single_view(img, base_name, output_dir, is_360)
            
            return output_paths
            
        except Exception as e:
            logger.error(f"360度画像処理中にエラー: {str(e)}")
            return []
    
    def _process_single_view(self, img, base_name, output_dir, is_360=True):
        """
        単一視点の画像を処理
        
        Args:
            img: OpenCV画像データ
            base_name: 出力ファイルのベース名
            output_dir: 出力ディレクトリ
            is_360: 360度画像かどうか
            
        Returns:
            処理された画像ファイルのパスのリスト
        """
        output_paths = []
        
        # 必要に応じてリサイズ
        if self.max_dimension:
            h, w = img.shape[:2]
            if max(h, w) > self.max_dimension:
                scale = self.max_dimension / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 全天球画像の場合、等距円筒図法から透視投影への変換
        if is_360 and self.equirect_to_perspective:
            # 6方向（キューブマップのような形式）に変換
            directions = [
                {"name": "front", "yaw": 0, "pitch": 0},
                {"name": "right", "yaw": 90, "pitch": 0},
                {"name": "back", "yaw": 180, "pitch": 0},
                {"name": "left", "yaw": 270, "pitch": 0},
                {"name": "up", "yaw": 0, "pitch": -90},
                {"name": "down", "yaw": 0, "pitch": 90}
            ]
            
            # 各方向ごとに透視投影画像を生成
            for direction in directions:
                perspective_img = self.equirectangular_to_perspective(
                    img, direction["yaw"], direction["pitch"])
                
                # 保存
                output_filename = f"{base_name}_{direction['name']}.{self.output_format}"
                output_path = os.path.join(output_dir, output_filename)
                
                if self.output_format in ['jpg', 'jpeg']:
                    cv2.imwrite(output_path, perspective_img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                elif self.output_format == 'png':
                    cv2.imwrite(output_path, perspective_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                else:
                    cv2.imwrite(output_path, perspective_img)
                
                output_paths.append(output_path)
        else:
            # 通常の画像として保存（360度でない場合や、変換を行わない設定の場合）
            output_filename = f"{base_name}.{self.output_format}"
            output_path = os.path.join(output_dir, output_filename)
            
            if self.output_format in ['jpg', 'jpeg']:
                cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            elif self.output_format == 'png':
                cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            else:
                cv2.imwrite(output_path, img)
            
            output_paths.append(output_path)
        
        return output_paths
    
    def equirectangular_to_perspective(self, img, yaw, pitch, fov=90):
        """
        等距円筒図法（全天球）から透視投影への変換
        
        Args:
            img: 全天球画像（等距円筒図法）
            yaw: 水平方向の角度（度）
            pitch: 垂直方向の角度（度）
            fov: 視野角（度）
            
        Returns:
            透視投影画像
        """
        h, w = img.shape[:2]
        
        # 出力サイズを設定（正方形）
        output_size = min(h, w) // 2
        
        # 角度をラジアンに変換
        yaw_rad = np.deg2rad(yaw)
        pitch_rad = np.deg2rad(pitch)
        fov_rad = np.deg2rad(fov)
        
        # 出力画像の座標グリッドを作成
        x = np.linspace(-np.tan(fov_rad/2), np.tan(fov_rad/2), output_size)
        y = np.linspace(-np.tan(fov_rad/2), np.tan(fov_rad/2), output_size)
        x_grid, y_grid = np.meshgrid(x, y)
        
        # 3D座標に変換
        xyz = np.zeros((output_size, output_size, 3))
        xyz[:, :, 0] = x_grid  # X座標
        xyz[:, :, 1] = y_grid  # Y座標
        xyz[:, :, 2] = 1.0     # Z座標（視点方向）
        
        # 回転行列を計算（ピッチとヨーの順に適用）
        # まずピッチ回転（X軸周り）
        rot_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])
        
        # 次にヨー回転（Y軸周り）
        rot_yaw = np.array([
            [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
            [0, 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
        ])
        
        # 回転行列を合成
        rot = rot_yaw @ rot_pitch
        
        # 各ピクセルの方向ベクトルを回転
        xyz_rotated = np.zeros_like(xyz)
        for i in range(output_size):
            for j in range(output_size):
                xyz_rotated[i, j] = rot @ xyz[i, j]
        
        # 球面座標に変換
        r = np.sqrt(np.sum(xyz_rotated**2, axis=2))
        theta = np.arccos(xyz_rotated[:, :, 2] / r)  # 極角（南北方向）
        phi = np.arctan2(xyz_rotated[:, :, 1], xyz_rotated[:, :, 0])  # 方位角（東西方向）
        
        # 等距円筒図法の座標に変換
        u = ((phi / (2 * np.pi)) % 1) * w
        v = (theta / np.pi) * h
        
        # 境界チェック
        u = np.clip(u, 0, w - 1).astype(int)
        v = np.clip(v, 0, h - 1).astype(int)
        
        # 出力画像を作成
        perspective = np.zeros((output_size, output_size, 3), dtype=np.uint8)
        
        # ピクセル値をサンプリング
        for i in range(output_size):
            for j in range(output_size):
                perspective[i, j] = img[v[i, j], u[i, j]]
        
        return perspective
    
    def process_directory(self, input_dir, output_dir, recursive=False, num_threads=None):
        """
        ディレクトリ内のすべての空間メディアファイルを処理
        
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
        all_extensions = self.insta360_extensions + self.standard_360_extensions + self.spatial_extensions
        
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
        
        # 通常のメディアと空間メディアを分離
        spatial_files = []
        
        for file_path in files_to_process:
            extension = file_path.suffix.lower()
            if extension in self.insta360_extensions:
                spatial_files.append(file_path)
            elif extension in self.standard_360_extensions:
                # メタデータで360かどうかを確認
                if self.is_360_media(file_path):
                    spatial_files.append(file_path)
            elif extension in self.spatial_extensions:
                spatial_files.append(file_path)
        
        total_files = len(spatial_files)
        logger.info(f"{total_files}個の空間メディアファイルが見つかりました")
        
        if total_files == 0:
            return []
        
        # 並列処理用のスレッド数
        num_threads = num_threads or min(os.cpu_count(), 8)
        logger.info(f"{num_threads}スレッドで処理を実行します")
        
        # 進捗カウンタ
        processed_count = 0
        
        # 各ファイルを処理する関数
        def process_file_with_progress(file_path):
            nonlocal processed_count
            result = self.process_file(file_path, output_dir)
            processed_count += 1
            if processed_count % 10 == 0 or processed_count == total_files:
                logger.info(f"進捗: {processed_count}/{total_files} ファイル処理済み")
            return result
        
        # 並列処理の実行
        all_processed_files = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(process_file_with_progress, spatial_files))
            for file_list in results:
                if file_list:
                    all_processed_files.extend(file_list)
        
        logger.info(f"処理完了: {len(all_processed_files)}個の画像ファイルが生成されました")
        return all_processed_files


def main():
    """コマンドラインからの実行用メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="空間撮影フォーマット処理ツール")
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
    parser.add_argument("--no-equirect", action="store_true", 
                        help="全天球画像から透視投影への変換を行わない")
    parser.add_argument("--no-stereo", action="store_true", 
                        help="ステレオ画像の分割を行わない")
    parser.add_argument("--threads", "-t", type=int, default=None, 
                        help="並列処理用のスレッド数")
    
    args = parser.parse_args()
    
    # 処理クラスの初期化
    processor = SpatialMediaProcessor(
        output_format=args.format,
        max_dimension=args.max_size,
        quality=args.quality,
        equirect_to_perspective=not args.no_equirect,
        extract_stereo=not args.no_stereo
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
            