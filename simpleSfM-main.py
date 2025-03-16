#!/usr/bin/env python3
"""
SimpleSfM - フォトグラメトリーアプリケーション
カメラ画像から3Dモデルを生成するためのGUIアプリケーション
"""

import os
import sys
import cv2
import numpy as np
import open3d as o3d
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                        QWidget, QPushButton, QFileDialog, QListWidget,
                        QLabel, QProgressBar, QMessageBox, QTabWidget, QSplitter,
                        QAction, QComboBox, QSpinBox, QCheckBox, QGroupBox,
                        QStatusBar, QMenu, QToolBar, QStyle)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QSize
from PyQt5.QtGui import QIcon, QPixmap
import pycolmap
from pathlib import Path
import gc
import logging
import json
import time
import psutil
import multiprocessing

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simpleSfM.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# アプリケーションの設定
APP_NAME = "SimpleSfM"
APP_VERSION = "1.0.0"

# モジュールのインポートチェック
try:
    from media_processor import MediaProcessor
    MEDIA_PROCESSOR_AVAILABLE = True
except ImportError:
    MEDIA_PROCESSOR_AVAILABLE = False
    logger.warning("MediaProcessorモジュールが見つかりません。RAW/WEBP/動画処理機能は無効化されます。")

try:
    from spatial_processor import SpatialMediaProcessor
    SPATIAL_PROCESSOR_AVAILABLE = True
except ImportError:
    SPATIAL_PROCESSOR_AVAILABLE = False
    logger.warning("SpatialMediaProcessorモジュールが見つかりません。空間メディア処理機能は無効化されます。")


class PhotogrammetryThread(QThread):
    """
    フォトグラメトリー処理を行うスレッド
    進捗状況を報告し、処理の完了を通知する
    """
    update_progress = pyqtSignal(int, str)
    process_complete = pyqtSignal(str)

    def __init__(self, image_dir, output_dir, sparse_model=True, dense_model=True, max_image_dimension=3000,
                num_threads=None, gpu_index=0, quality='medium'):
        super().__init__()
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.sparse_model = sparse_model
        self.dense_model = dense_model
        self.max_image_dimension = max_image_dimension  # 画像サイズ制限
        self.num_threads = num_threads or os.cpu_count()  # CPUスレッド数
        self.gpu_index = gpu_index  # GPU使用設定
        self.quality = quality  # 品質設定（'low', 'medium', 'high'）

    def preprocess_images(self, input_dir, output_dir):
        """大量の画像を前処理して、サイズを縮小しメモリ使用量を抑える"""
        os.makedirs(output_dir, exist_ok=True)
        self.update_progress.emit(5, "画像を前処理中...")

        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(list(Path(input_dir).glob(f'*{ext}')))
            image_paths.extend(list(Path(input_dir).glob(f'*{ext.upper()}')))

        total_images = len(image_paths)
        if total_images == 0:
            raise Exception("画像が見つかりませんでした。")

        for i, img_path in enumerate(image_paths):
            if i % max(1, total_images // 20) == 0:  # 進捗を20段階で表示
                self.update_progress.emit(5 + (i * 5) // total_images, f"画像を処理中... ({i+1}/{total_images})")

            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # 画像のリサイズ
                h, w = img.shape[:2]
                if max(h, w) > self.max_image_dimension:
                    scale = self.max_image_dimension / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # 処理済み画像を保存
                out_path = os.path.join(output_dir, img_path.name)
                cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            except Exception as e:
                logger.error(f"画像の処理中にエラー: {str(e)} - {img_path}")

        return output_dir

    def run(self):
        try:
            # 出力ディレクトリを作成
            os.makedirs(self.output_dir, exist_ok=True)

            # 画像の前処理（オプション）
            preprocessed_dir = os.path.join(self.output_dir, "preprocessed_images")
            processed_image_dir = self.preprocess_images(self.image_dir, preprocessed_dir)

            # COLMAP SfM処理の設定
            self.update_progress.emit(10, "画像をロード中...")

            # カメラデータベースの作成
            database_path = os.path.join(self.output_dir, "database.db")

            # 特徴抽出オプションを設定
            extraction_options = {
                "database_path": database_path,
                "image_path": processed_image_dir,
                "num_threads": self.num_threads
            }

            # 品質に基づく設定
            if self.quality == 'low':
                extraction_options["sift_options"] = {"max_num_features": 3000}
            elif self.quality == 'high':
                extraction_options["sift_options"] = {"max_num_features": 8000}
            else:  # medium
                extraction_options["sift_options"] = {"max_num_features": 5000}

            # 特徴抽出（バッチ処理）
            self.update_progress.emit(20, "特徴点を抽出中...")
            pycolmap.extract_features(**extraction_options)

            # 特徴マッチング（バッチ処理とマルチスレッド化）
            self.update_progress.emit(30, "特徴点をマッチング中...")

            # 総当たりマッチングではなく、ボキャブラリーツリーマッチングを使用
            # これは大量画像に対して効率的
            matcher_options = {
                "database_path": database_path,
                "num_threads": self.num_threads
            }

            # 画像数が多い場合は、スケーラブルなマッチング方法を選択
            import glob
            image_count = len(glob.glob(os.path.join(processed_image_dir, "*.[jJ][pP][gG]"))) + \
                        len(glob.glob(os.path.join(processed_image_dir, "*.[jJ][pP][eE][gG]"))) + \
                        len(glob.glob(os.path.join(processed_image_dir, "*.[pP][nN][gG]"))) + \
                        len(glob.glob(os.path.join(processed_image_dir, "*.[tT][iI][fF]"))) + \
                        len(glob.glob(os.path.join(processed_image_dir, "*.[tT][iI][fF][fF]")))

            if image_count > 100:
                self.update_progress.emit(35, "ボキャブラリーツリーマッチングを使用中...")
                pycolmap.match_sequential(**matcher_options)
            else:
                pycolmap.match_exhaustive(**matcher_options)

            if self.sparse_model:
                # スパース再構成
                self.update_progress.emit(50, "スパース再構成を実行中...")
                sparse_dir = os.path.join(self.output_dir, "sparse")
                os.makedirs(sparse_dir, exist_ok=True)

                mapper_options = {
                    "database_path": database_path,
                    "image_path": processed_image_dir,
                    "output_path": sparse_dir,
                }

                reconstructions = pycolmap.incremental_mapping(**mapper_options)

                if not reconstructions:
                    self.process_complete.emit("スパース再構成に失敗しました。")
                    return

                self.update_progress.emit(70, "スパース再構成完了")

            if self.dense_model and self.sparse_model:
                # デンス再構成
                self.update_progress.emit(75, "デンス再構成を実行中...")
                dense_dir = os.path.join(self.output_dir, "dense")
                os.makedirs(dense_dir, exist_ok=True)

                # 画像のアンディストーション
                self.update_progress.emit(80, "画像を正規化中...")
                pycolmap.undistort_images(sparse_dir, processed_image_dir, dense_dir)

                # メモリ使用量を制限するために各ステップで不要なデータを解放
                import gc
                gc.collect()

                # MVSによるデンス点群生成（内部的にはバッチ処理を採用）
                self.update_progress.emit(85, "デンス点群を生成中...")
                stereo_options = {}

                if self.quality == 'low':
                    stereo_options = {"max_image_size": 1000, "window_radius": 3}
                elif self.quality == 'high':
                    stereo_options = {"max_image_size": 2000, "window_radius": 7}
                else:  # medium
                    stereo_options = {"max_image_size": 1500, "window_radius": 5}

                pycolmap.patch_match_stereo(dense_dir, **stereo_options)

                # デンス点群の融合（メモリ効率化）
                self.update_progress.emit(90, "点群を融合中...")

                fusion_options = {}
                if self.quality == 'low':
                    fusion_options = {"min_num_pixels": 3, "max_reproj_error": 4}
                elif self.quality == 'high':
                    fusion_options = {"min_num_pixels": 5, "max_reproj_error": 2}

                pycolmap.stereo_fusion(os.path.join(dense_dir, "fused.ply"), dense_dir, **fusion_options)

                # メモリを解放
                gc.collect()

                # メッシュ生成（Open3Dを使用、メモリ使用量最適化）
                self.update_progress.emit(95, "メッシュを生成中...")
                try:
                    # PLYファイルから点群を読み込み
                    ply_path = os.path.join(dense_dir, "fused.ply")

                    # ダウンサンプリングも検討
                    point_cloud = o3d.io.read_point_cloud(ply_path)
                    original_points = len(point_cloud.points)

                    # 巨大な点群の場合はダウンサンプリング
                    if original_points > 1000000:  # 100万点以上の場合
                        self.update_progress.emit(96, f"点群をダウンサンプリング中... ({original_points}点)")
                        point_cloud = point_cloud.voxel_down_sample(voxel_size=0.01)
                        self.update_progress.emit(97, f"ダウンサンプリング完了 ({len(point_cloud.points)}点)")

                    # 法線を推定
                    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.05, max_nn=30))

                    # ポアソン再構成のパラメータも品質設定に応じて調整
                    poisson_depth = 8
                    if self.quality == 'low':
                        poisson_depth = 7
                    elif self.quality == 'high':
                        poisson_depth = 9

                    # メッシュ生成
                    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        point_cloud, depth=poisson_depth, width=0, scale=1.1, linear_fit=False)

                    # メッシュの後処理（小さな接続されていない部分を削除）
                    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
                    triangle_clusters = np.asarray(triangle_clusters)
                    cluster_n_triangles = np.asarray(cluster_n_triangles)
                    if len(cluster_n_triangles) > 0:
                        largest_cluster_idx = np.argmax(cluster_n_triangles)
                        triangles_to_remove = triangle_clusters != largest_cluster_idx
                        mesh.remove_triangles_by_mask(triangles_to_remove)

                    # メッシュを保存
                    o3d.io.write_triangle_mesh(os.path.join(self.output_dir, "mesh.obj"), mesh)

                    # 低解像度版も作成（ビューワー用）
                    simplified_mesh = mesh.simplify_quadric_decimation(
                        target_number_of_triangles=min(100000, len(mesh.triangles)))
                    o3d.io.write_triangle_mesh(os.path.join(self.output_dir, "mesh_simplified.obj"), simplified_mesh)

                except Exception as e:
                    self.process_complete.emit(f"メッシュ生成中にエラーが発生しました: {str(e)}")
                    return

            self.update_progress.emit(100, "処理完了")
            self.process_complete.emit("3Dモデルの生成が完了しました！")

        except Exception as e:
            self.process_complete.emit(f"エラーが発生しました: {str(e)}")


class ModelViewerWidget(QWidget):
    """
    3Dモデルビューワーウィジェット
    シンプルなモデル表示と制御機能を提供
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.loaded_model_path = None
        self.vis = None

        # コントロールパネル
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)

        # ビューモード
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["シェード", "ワイヤーフレーム", "テクスチャ", "点群"])
        self.view_mode_combo.currentIndexChanged.connect(self.change_view_mode)
        control_layout.addWidget(QLabel("表示モード:"))
        control_layout.addWidget(self.view_mode_combo)

        # 分割線
        control_layout.addWidget(QLabel("|"))

        # カメラ制御
        self.reset_view_btn = QPushButton("ビューをリセット")
        self.reset_view_btn.clicked.connect(self.reset_view)
        control_layout.addWidget(self.reset_view_btn)

        # エクスポートボタン
        self.export_btn = QPushButton("エクスポート")
        self.export_btn.clicked.connect(self.export_model)
        control_layout.addWidget(self.export_btn)

        # 表示領域（初期状態）
        self.viewer_frame = QWidget()
        self.viewer_frame.setLayout(QVBoxLayout())
        self.viewer_frame.setFrameShape(QWidget.StyledPanel)
        self.viewer_frame.setMinimumHeight(400)

        # 初期テキスト
        self.info_label = QLabel("モデルが読み込まれていません")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.viewer_frame.layout().addWidget(self.info_label)

        # 統計情報表示
        self.stats_label = QLabel("")

        # レイアウトに追加
        self.layout.addWidget(control_panel)
        self.layout.addWidget(self.viewer_frame)
        self.layout.addWidget(self.stats_label)

        # オープンソースの3Dビューワーへのリンク
        info_text = "注: 高度な3D操作には、以下のアプリケーションへのエクスポートをお勧めします: MeshLab, Blender, ..."
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        self.layout.addWidget(info_label)

    def initialize_viewer(self):
        """Open3D可視化ウィンドウを初期化"""
        try:
            # もし既存のビジュアライザがあれば閉じる
            if self.vis:
                self.vis.close()

            # Open3Dのビジュアライザを初期化
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(width=800, height=600)

            # レンダリング設定
            render_option = self.vis.get_render_option()
            render_option.point_size = 2
            render_option.background_color = [0.8, 0.8, 0.8]  # 明るいグレー

            return True
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"ビューワーの初期化に失敗しました: {str(e)}")
            return False

    def load_model(self, model_path):
        try:
            self.loaded_model_path = model_path
            self.info_label.setText(f"モデルをロード中: {os.path.basename(model_path)}")

            # 本格的な3D表示のためのコードを実装（簡易版）
            if not self.initialize_viewer():
                return

            # 簡易表示または詳細表示
            try:
                # メッシュファイルの読み込み
                if model_path.endswith('.obj') or model_path.endswith('.ply'):
                    mesh = o3d.io.read_triangle_mesh(model_path)

                    # 法線が無い場合は計算
                    if not mesh.has_vertex_normals():
                        mesh.compute_vertex_normals()

                    # 統計情報の更新
                    vertices = len(mesh.vertices)
                    triangles = len(mesh.triangles)
                    self.stats_label.setText(f"統計: 頂点数 {vertices:,}個 / 三角形数 {triangles:,}個")

                    # ビジュアライザに追加
                    self.vis.add_geometry(mesh)

                    # カメラ位置の最適化
                    self.vis.reset_view_point(True)
                    self.vis.update_renderer()

                    # ウィンドウのスクリーンショットを取得
                    screenshot_path = os.path.join(os.path.dirname(model_path), "preview.png")
                    self.vis.capture_screen_image(screenshot_path)

                    # 画像を表示
                    pixmap = QPixmap(screenshot_path)
                    scaled_pixmap = pixmap.scaled(self.viewer_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

                    # 既存のラベルをクリア
                    for i in reversed(range(self.viewer_frame.layout().count())):
                        self.viewer_frame.layout().itemAt(i).widget().deleteLater()

                    # 新しい画像ラベルを作成
                    image_label = QLabel()
                    image_label.setPixmap(scaled_pixmap)
                    image_label.setAlignment(Qt.AlignCenter)
                    self.viewer_frame.layout().addWidget(image_label)

                elif model_path.endswith('.ply'):  # 点群の場合
                    pcd = o3d.io.read_point_cloud(model_path)

                    # 統計情報の更新
                    points = len(pcd.points)
                    self.stats_label.setText(f"統計: 点数 {points:,}個")

                    # ビジュアライザに追加
                    self.vis.add_geometry(pcd)

                    # 以下同様にスクリーンショット取得と表示
                    self.vis.reset_view_point(True)
                    self.vis.update_renderer()

                    screenshot_path = os.path.join(os.path.dirname(model_path), "preview.png")
                    self.vis.capture_screen_image(screenshot_path)

                    pixmap = QPixmap(screenshot_path)
                    scaled_pixmap = pixmap.scaled(self.viewer_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

                    for i in reversed(range(self.viewer_frame.layout().count())):
                        self.viewer_frame.layout().itemAt(i).widget().deleteLater()

                    image_label = QLabel()
                    image_label.setPixmap(scaled_pixmap)
                    image_label.setAlignment(Qt.AlignCenter)
                    self.viewer_frame.layout().addWidget(image_label)

                else:
                    QMessageBox.warning(self, "警告", f"未対応のファイル形式です: {os.path.basename(model_path)}")

            except Exception as e:
                QMessageBox.warning(self, "警告", f"モデルを表示できません: {str(e)}")
                self.info_label.setText(f"モデルの表示に失敗しました: {os.path.basename(model_path)}")

        except Exception as e:
            QMessageBox.critical(self, "エラー", f"モデルのロード中にエラーが発生しました: {str(e)}")
            self.info_label.setText("モデルのロードに失敗しました")

    def change_view_mode(self, index):
        """表示モードの変更"""
        if not self.vis or not self.loaded_model_path:
            return

        # レンダリングオプションの取得
        render_option = self.vis.get_render_option()

        # インデックスに基づいて表示モードを設定
        if index == 0:  # シェード
            render_option.mesh_show_wireframe = False
            render_option.mesh_show_back_face = False
        elif index == 1:  # ワイヤーフレーム
            render_option.mesh_show_wireframe = True
            render_option.line_width = 1.0
        elif index == 2:  # テクスチャ
            render_option.mesh_show_wireframe = False
            render_option.mesh_show_back_face = True
        elif index == 3:  # 点群
            # 点群モードの実装
            pass

        # ビューを更新してスクリーンショットを取得
        self.vis.update_renderer()
        self.update_viewer_image()

    def reset_view(self):
        """カメラ位置のリセット"""
        if not self.vis:
            return

        self.vis.reset_view_point(True)
        self.vis.update_renderer()
        self.update_viewer_image()

    def update_viewer_image(self):
        """ビューワー画像を更新"""
        if not self.vis or not self.loaded_model_path:
            return

        try:
            # スクリーンショットを取得
            screenshot_path = os.path.join(os.path.dirname(self.loaded_model_path), "preview.png")
            self.vis.capture_screen_image(screenshot_path)

            # 画像を表示
            pixmap = QPixmap(screenshot_path)
            scaled_pixmap = pixmap.scaled(self.viewer_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # 既存のラベルを更新
            for i in range(self.viewer_frame.layout().count()):
                widget = self.viewer_frame.layout().itemAt(i).widget()
                if isinstance(widget, QLabel) and widget.pixmap():
                    widget.setPixmap(scaled_pixmap)
                    break
        except Exception as e:
            logger.error(f"ビューワー画像更新エラー: {str(e)}")

    def export_model(self):
        """モデルのエクスポート"""
        if not self.loaded_model_path:
            QMessageBox.warning(self, "警告", "エクスポートするモデルがありません")
            return

        # エクスポート形式の選択
        formats = ["Wavefront OBJ (*.obj)", "PLY (*.ply)", "STL (*.stl)"]
        export_format, ok = QFileDialog.getSaveFileName(
            self, "モデルをエクスポート", "", ";;".join(formats))

        if not export_format:
            return

        try:
            # 入力モデルの読み込み
            if self.loaded_model_path.endswith('.obj'):
                mesh = o3d.io.read_triangle_mesh(self.loaded_model_path)
            elif self.loaded_model_path.endswith('.ply'):
                # PLYファイルがメッシュか点群かを判定
                try:
                    mesh = o3d.io.read_triangle_mesh(self.loaded_model_path)
                    if len(mesh.triangles) == 0:
                        pcd = o3d.io.read_point_cloud(self.loaded_model_path)
                        # 点群からメッシュを生成
                        pcd.estimate_normals()
                        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
                except:
                    pcd = o3d.io.read_point_cloud(self.loaded_model_path)
                    pcd.estimate_normals()
                    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

            # 出力形式に応じてエクスポート
            if export_format.endswith('.obj'):
                o3d.io.write_triangle_mesh(export_format, mesh)
            elif export_format.endswith('.ply'):
                o3d.io.write_triangle_mesh(export_format, mesh)
            elif export_format.endswith('.stl'):
                o3d.io.write_triangle_mesh(export_format, mesh)

            QMessageBox.information(self, "エクスポート成功", f"モデルを正常にエクスポートしました: {export_format}")

        except Exception as e:
            QMessageBox.critical(self, "エラー", f"エクスポート中にエラーが発生しました: {str(e)}")

    def resizeEvent(self, event):
        """ウィンドウサイズ変更時に表示を更新"""
        super().resizeEvent(event)
        self.update_viewer_image()


class MainWindow(QMainWindow):
    """
    メインウィンドウクラス
    アプリケーションのメインUIとロジックを提供
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1000, 700)

        self.image_dir = None
        self.output_dir = None
        self.processing_thread = None

        # 高度なパラメータの初期値
        self.max_image_dimension = 3000
        self.num_threads = os.cpu_count()
        self.gpu_index = 0
        self.quality = 'medium'

        # 設定の読み込み
        self.settings = QSettings("SimpleSfM", "Settings")
        self.load_settings()

        # UIの初期化
        self.init_ui()
        self.init_menu()
        self.update_ui_state()

    def load_settings(self):
        """アプリケーション設定の読み込み"""
        self.max_image_dimension = self.settings.value("max_image_dimension", 3000, type=int)
        self.num_threads = self.settings.value("num_threads", os.cpu_count(), type=int)
        self.quality = self.settings.value("quality", "medium", type=str)
        self.last_dir = self.settings.value("last_dir", "", type=str)

    def save_settings(self):
        """アプリケーション設定の保存"""
        self.settings.setValue("max_image_dimension", self.max_image_dimension)
        self.settings.setValue("num_threads", self.num_threads)
        self.settings.setValue("quality", self.quality)
        if self.image_dir:
            self.settings.setValue("last_dir", self.image_dir)

    def init_ui(self):
        """メインUIの初期化"""
        # メインウィジェットとレイアウト
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # タブウィジェット
        self.tabs = QTabWidget()

        # ワークフロータブ
        workflow_tab = QWidget()
        workflow_layout = QVBoxLayout(workflow_tab)

        # 画像セクション
        image_section = QWidget()
        image_layout = QVBoxLayout(image_section)

        # 画像読み込みボタン
        self.load_images_btn = QPushButton("画像フォルダを選択")
        self.load_images_btn.clicked.connect(self.load_images)
        image_layout.addWidget(self.load_images_btn)

        # 現在の状態表示
        self.status_label = QLabel("ステータス: 画像が読み込まれていません")
        image_layout.addWidget(self.status_label)

        # 画像リスト
        self.image_list = QListWidget()
        image_layout.addWidget(self.image_list)

        # 処理セクション
        process_section = QWidget()
        process_layout = QVBoxLayout(process_section)

        # 出力フォルダ選択
        output_layout = QHBoxLayout()
        self.output_label = QLabel("出力フォルダ: 未選択")
        self.select_output_btn = QPushButton("選択")
        self.select_output_btn.clicked.connect(self.select_output_dir)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.select_output_btn)
        process_layout.addLayout(output_layout)

        # 高度な設定セクション
        advanced_group = QWidget()
        advanced_layout = QVBoxLayout(advanced_group)
        advanced_layout.setContentsMargins(0, 0, 0, 0)

        # 品質選択
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("処理品質:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(['低 (速い)', '中 (バランス)', '高 (遅い)'])
        # 現在の品質設定に基づいて選択
        if self.quality == 'low':
            self.quality_combo.setCurrentIndex(0)
        elif self.quality == 'high':
            self.quality_combo.setCurrentIndex(2)
        else:
            self.quality_combo.setCurrentIndex(1)
        self.quality_combo.currentIndexChanged.connect(self.update_quality)
        quality_layout.addWidget(self.quality_combo)
        advanced_layout.addLayout(quality_layout)

        # 最大画像サイズ
        img_size_layout = QHBoxLayout()
        img_size_layout.addWidget(QLabel("最大画像サイズ:"))
        self.img_size_combo = QComboBox()
        self.img_size_combo.addItems(['1000px', '2000px', '3000px', '4000px', '元のサイズ'])
        # 現在のサイズ設定に基づいて選択
        if self.max_image_dimension == 1000:
            self.img_size_combo.setCurrentIndex(0)
        elif self.max_image_dimension == 2000:
            self.img_size_combo.setCurrentIndex(1)
        elif self.max_image_dimension == 3000:
            self.img_size_combo.setCurrentIndex(2)
        elif self.max_image_dimension == 4000:
            self.img_size_combo.setCurrentIndex(3)
        else:
            self.img_size_combo.setCurrentIndex(4)
        self.img_size_combo.currentIndexChanged.connect(self.update_image_size)
        img_size_layout.addWidget(self.img_size_combo)
        advanced_layout.addLayout(img_size_layout)

        # CPU スレッド数
        thread_layout = QHBoxLayout()
        thread_layout.addWidget(QLabel("使用CPUスレッド:"))
        self.thread_combo = QComboBox()
        max_threads = os.cpu_count() or 4
        thread_options = ['1', '2', '4', str(max(1, max_threads // 2)), str(max_threads)]
        # 重複を除去
        thread_options = sorted(list(set([int(x) for x in thread_options])))
        self.thread_combo.addItems([str(x) for x in thread_options])
        # 現在のスレッド設定に基づいて選択
        current_thread_index = 0
        for i, threads in enumerate(thread_options):
            if threads == self.num_threads:
                current_thread_index = i
                break
        self.thread_combo.setCurrentIndex(current_thread_index)
        self.thread_combo.currentIndexChanged.connect(self.update_threads)
        thread_layout.addWidget(self.thread_combo)
        advanced_layout.addLayout(thread_layout)

        # 高度な設定を追加
        process_layout.addWidget(advanced_group)

        # 処理ボタン
        button_layout = QHBoxLayout()
        self.align_photos_btn = QPushButton("画像をアライン")
        self.align_photos_btn.clicked.connect(lambda: self.start_processing(sparse_model=True, dense_model=False))

        self.build_dense_cloud_btn = QPushButton("デンス点群を構築")
        self.build_dense_cloud_btn.clicked.connect(lambda: self.start_processing(sparse_model=False, dense_model=True))

        self.build_model_btn = QPushButton("3Dモデルを生成")
        self.build_model_btn.clicked.connect(lambda: self.start_processing(sparse_model=True, dense_model=True))

        button_layout.addWidget(self.align_photos_btn)
        button_layout.addWidget(self.build_dense_cloud_btn)
        button_layout.addWidget(self.build_model_btn)
        process_layout.addLayout(button_layout)

        # 計算リソース使用量予測
        self.resource_label = QLabel("推定メモリ使用量: --")
        process_layout.addWidget(self.resource_label)

        # プログレスバー
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("準備完了")
        process_layout.addWidget(self.progress_bar)
        process_layout.addWidget(self.progress_label)

        # キャンセルボタン
        self.cancel_btn = QPushButton("処理をキャンセル")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        process_layout.addWidget(self.cancel_btn)

        # ワークフローレイアウトに追加
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(image_section)
        splitter.addWidget(process_section)
        workflow_layout.addWidget(splitter)

        # モデルビュータブ
        self.model_viewer = ModelViewerWidget()

        # 設定タブ
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)

        # 詳細設定
        settings_group = QGroupBox("詳細設定")
        settings_group_layout = QVBoxLayout()

        # 処理オプション
        processing_options_group = QGroupBox("処理オプション")
        processing_options_layout = QVBoxLayout()

        # GPUオプション
        gpu_check = QCheckBox("GPUを使用する（可能な場合）")
        gpu_check.setChecked(True)
        processing_options_layout.addWidget(gpu_check)

        # 高度なオプションなど...

        processing_options_group.setLayout(processing_options_layout)
        settings_group_layout.addWidget(processing_options_group)

        # 出力オプション
        output_options_group = QGroupBox("出力オプション")
        output_options_layout = QVBoxLayout()

        # テクスチャオプション
        texture_check = QCheckBox("テクスチャを生成")
        texture_check.setChecked(True)
        output_options_layout.addWidget(texture_check)

        # 簡略化オプション
        simplify_check = QCheckBox("メッシュを自動簡略化")
        simplify_check.setChecked(True)
        output_options_layout.addWidget(simplify_check)

        output_options_group.setLayout(output_options_layout)
        settings_group_layout.addWidget(output_options_group)

        settings_group.setLayout(settings_group_layout)
        settings_layout.addWidget(settings_group)
        settings_layout.addStretch(1)

        # タブに追加
        self.tabs.addTab(workflow_tab, "ワークフロー")
        self.tabs.addTab(self.model_viewer, "モデルビュー")
        self.tabs.addTab(settings_tab, "詳細設定")

        main_layout.addWidget(self.tabs)
        self.setCentralWidget(main_widget)

        # ステータスバー
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("準備完了")

    def init_menu(self):
        """メニューとツールバーの初期化"""
        # メインメニュー
        menubar = self.menuBar()

        # ファイルメニュー
        file_menu = menubar.addMenu("ファイル(&F)")

        # 画像読み込みアクション
        load_images_action = QAction("画像フォルダを開く...", self)
        load_images_action.setShortcut("Ctrl+O")
        load_images_action.triggered.connect(self.load_images)
        file_menu.addAction(load_images_action)

        # 拡張メディアサポート（利用可能な場合）
        if MEDIA_PROCESSOR_AVAILABLE:
            self.init_media_support(file_menu)

        # 空間メディアサポート（利用可能な場合）
        if SPATIAL_PROCESSOR_AVAILABLE:
            self.init_spatial_media_support()

        file_menu.addSeparator()

        # エクスポートアクション
        export_action = QAction("モデルをエクスポート...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_model)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        # 終了アクション
        exit_action = QAction("終了", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 編集メニュー
        edit_menu = menubar.addMenu("編集(&E)")

        # 設定アクション
        settings_action = QAction("設定...", self)
        settings_action.triggered.connect(lambda: self.tabs.setCurrentIndex(2))
        edit_menu.addAction(settings_action)

        # ヘルプメニュー
        help_menu = menubar.addMenu("ヘルプ(&H)")

        # マニュアルアクション
        manual_action = QAction("マニュアル", self)
        manual_action.triggered.connect(self.show_manual)
        help_menu.addAction(manual_action)

        # バージョン情報アクション
        about_action = QAction("バージョン情報", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        # ツールバー
        toolbar = self.addToolBar("メインツールバー")
        toolbar.addAction(load_images_action)
        toolbar.addAction(export_action)

    def init_media_support(self, file_menu=None):
        """メディア処理サポートの初期化"""
        if file_menu is None:
            file_menu = self.menuBar().addMenu("ファイル(&F)")

        # メディアインポートアクション
        import_action = QAction("RAW/WEBP/動画をインポート...", self)
        import_action.setStatusTip("RAWファイル、WEBPファイル、動画ファイルを処理して取り込む")
        import_action.triggered.connect(self.show_media_import_dialog)

        file_menu.addAction(import_action)

        # ツールバーに追加
        media_toolbar = self.addToolBar("メディア")
        media_toolbar.addAction(import_action)

    def init_spatial_media_support(self):
        """空間メディア処理サポートの初期化"""
        # ツールバーに追加
        media_toolbar = self.findChild(QToolBar, "メディア") or self.addToolBar("メディア")

        # 空間メディアインポートアクション
        spatial_import_action = QAction("空間メディアをインポート...", self)
        spatial_import_action.setStatusTip("Insta360やXREALなどの空間メディアを処理して取り込む")
        spatial_import_action.triggered.connect(self.show_spatial_media_import_dialog)

        media_toolbar.addAction(spatial_import_action)

        # メインメニューにも追加
        media_menu = self.menuBar().findChild(QMenu, "メディア(&M)")
        if not media_menu:
            media_menu = self.menuBar().addMenu("メディア(&M)")
        media_menu.addAction(spatial_import_action)

    def show_media_import_dialog(self):
        """メディアインポートダイアログを表示"""
        if not MEDIA_PROCESSOR_AVAILABLE:
            QMessageBox.warning(self, "未対応", "RAW/WEBP/動画処理モジュールが見つかりません。")
            return

        from media_dialog import MediaImportDialog
        import_dialog = MediaImportDialog(self)
        import_dialog.import_complete.connect(self.on_media_import_complete)
        import_dialog.setWindowModality(Qt.ApplicationModal)  # モーダルダイアログとして表示
        import_dialog.show()

    def show_spatial_media_import_dialog(self):
        """空間メディアインポートダイアログを表示"""
        if not SPATIAL_PROCESSOR_AVAILABLE:
            QMessageBox.warning(self, "未対応", "空間メディア処理モジュールが見つかりません。")
            return

        from spatial_media_dialog import SpatialMediaImportDialog
        spatial_dialog = SpatialMediaImportDialog(self)
        spatial_dialog.import_complete.connect(self.on_spatial_media_import_complete)
        spatial_dialog.setWindowModality(Qt.ApplicationModal)  # モーダルダイアログとして表示
        spatial_dialog.show()

    def on_media_import_complete(self, processed_dir):
        """メディアインポート完了時の処理"""
        if processed_dir and os.path.exists(processed_dir):
            # 処理済みディレクトリを現在の画像ディレクトリとして設定
            self.load_images_from_dir(processed_dir)

    def on_spatial_media_import_complete(self, processed_dir):
        """空間メディアインポート完了時の処理"""
        if processed_dir and os.path.exists(processed_dir):
            # 処理済みディレクトリを現在の画像ディレクトリとして設定
            self.load_images_from_dir(processed_dir)

            # 追加情報を表示
            QMessageBox.information(self, "空間メディア読み込み完了",
                                f"空間メディアから抽出された画像を読み込みました。\n\n"
                                f"ディレクトリ: {processed_dir}\n\n"
                                f"これらの画像を使用して3Dモデルを生成できます。")

    def update_quality(self, index):
        """品質設定の更新"""
        quality_map = {0: 'low', 1: 'medium', 2: 'high'}
        self.quality = quality_map.get(index, 'medium')
        self.update_resource_estimate()

    def update_image_size(self, index):
        """最大画像サイズの更新"""
        size_map = {0: 1000, 1: 2000, 2: 3000, 3: 4000, 4: None}  # None は制限なし
        self.max_image_dimension = size_map.get(index, 3000)
        self.update_resource_estimate()

    def update_threads(self, index):
        """スレッド数の更新"""
        self.num_threads = int(self.thread_combo.currentText())
        self.update_resource_estimate()

    def update_resource_estimate(self):
        """メモリ使用量と処理時間の予測"""
        if not self.image_dir:
            return

        try:
            # 画像数をカウント
            image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
            image_count = 0
            for ext in image_extensions:
                image_count += len(list(Path(self.image_dir).glob(f'*{ext}')))
                image_count += len(list(Path(self.image_dir).glob(f'*{ext.upper()}')))

            if image_count == 0:
                self.resource_label.setText("画像が見つかりません")
                return

            # 品質ベースの係数
            quality_factor = {'low': 0.5, 'medium': 1.0, 'high': 2.0}[self.quality]

            # サイズベースの係数
            if self.max_image_dimension is None:
                size_factor = 2.0  # 制限なしの場合は高めに見積もる
            else:
                size_factor = (self.max_image_dimension / 3000.0) ** 2

            # 基本メモリ使用量の見積もり（1枚あたり）
            per_image_memory_mb = 250 * quality_factor * size_factor

            # 総メモリ見積もり
            total_memory_gb = (per_image_memory_mb * image_count) / 1024

            # スレッドによる並列性の考慮
            effective_memory_gb = min(total_memory_gb, total_memory_gb * self.num_threads / 4)

            # 処理時間の見積もり
            time_factor = {'low': 0.5, 'medium': 1.0, 'high': 2.5}[self.quality]

            # 処理時間はサイズ、画像数、スレッド数による
            basic_time_minutes = (image_count * 0.5) * time_factor * (size_factor ** 0.7)
            thread_speedup = min(1.0, self.num_threads / 4) * 0.8 + 0.2  # スレッドによる高速化（ある程度まで）
            estimated_time = basic_time_minutes / thread_speedup

            # 時間の単位変換
            if estimated_time < 60:
                time_str = f"{estimated_time:.1f}分"
            else:
                time_str = f"{estimated_time/60:.1f}時間"

            # 表示更新
            self.resource_label.setText(
                f"推定メモリ使用量: {effective_memory_gb:.1f}GB / 推定処理時間: {time_str} / 画像数: {image_count}枚"
            )

        except Exception as e:
            logger.error(f"リソース見積もりエラー: {str(e)}")
            self.resource_label.setText("リソース推定: 計算できません")

    def load_images(self):
        """画像フォルダを選択して読み込む"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "画像フォルダを選択",
            self.last_dir if self.last_dir else ""
        )

        if dir_path:
            self.load_images_from_dir(dir_path)

    def load_images_from_dir(self, dir_path):
        """指定ディレクトリから画像を読み込み"""
        if os.path.exists(dir_path):
            self.image_dir = dir_path
            self.last_dir = dir_path
            self.image_list.clear()

            # フォルダ内の画像ファイルをリストアップ
            image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
            image_files = []

            for ext in image_extensions:
                image_files.extend(list(Path(dir_path).glob(f'*{ext}')))
                image_files.extend(list(Path(dir_path).glob(f'*{ext.upper()}')))

            # 大量の画像の場合はサンプル表示
            max_display = 1000
            total_images = len(image_files)

            if total_images > max_display:
                # サンプルを選択
                image_files = image_files[:max_display]
                self.status_label.setText(f"ステータス: {total_images}枚の画像が読み込まれました（一部のみ表示）")
            else:
                self.status_label.setText(f"ステータス: {total_images}枚の画像が読み込まれました")

            # リストに追加
            for img_path in image_files:
                self.image_list.addItem(img_path.name)

            self.update_ui_state()
            self.update_resource_estimate()

            # 出力ディレクトリのデフォルト値を設定
            default_output = os.path.join(dir_path, "output")
            self.output_dir = default_output
            self.output_label.setText(f"出力フォルダ: {default_output}")

    def select_output_dir(self):
        """出力フォルダを選択"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "出力フォルダを選択",
            self.output_dir if self.output_dir else ""
        )

        if dir_path:
            self.output_dir = dir_path
            self.output_label.setText(f"出力フォルダ: {dir_path}")
            self.update_ui_state()

    def start_processing(self, sparse_model=True, dense_model=True):
        """フォトグラメトリー処理を開始"""
        if not self.image_dir or not self.output_dir:
            QMessageBox.warning(self, "警告", "画像フォルダと出力フォルダを選択してください。")
            return

        # UIを無効化
        self.set_ui_enabled(False)
        self.cancel_btn.setEnabled(True)

        # プログレスバーをリセット
        self.progress_bar.setValue(0)
        self.progress_label.setText("処理を開始します...")

        # 品質文字列マッピング
        quality_map = {0: 'low', 1: 'medium', 2: 'high'}
        quality_idx = self.quality_combo.currentIndex()
        quality = quality_map.get(quality_idx, 'medium')

        # 処理スレッドを開始
        self.processing_thread = PhotogrammetryThread(
            self.image_dir, self.output_dir, sparse_model, dense_model,
            max_image_dimension=self.max_image_dimension,
            num_threads=self.num_threads,
            gpu_index=self.gpu_index,
            quality=self.quality
        )
        self.processing_thread.update_progress.connect(self.update_progress)
        self.processing_thread.process_complete.connect(self.process_complete)
        self.processing_thread.start()

    def cancel_processing(self):
        """処理をキャンセル"""
        if self.processing_thread and self.processing_thread.isRunning():
            # スレッドを強制終了
            self.processing_thread.terminate()
            self.processing_thread.wait()

            # UI状態を更新
            self.progress_label.setText("処理がキャンセルされました")
            self.set_ui_enabled(True)
            self.cancel_btn.setEnabled(False)

            QMessageBox.information(self, "キャンセル", "処理がキャンセルされました。")

    def update_progress(self, value, message):
        """進捗の更新"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
        self.statusBar.showMessage(message)

    def process_complete(self, message):
        """処理完了のコールバック"""
        self.progress_label.setText(message)
        self.set_ui_enabled(True)
        self.cancel_btn.setEnabled(False)

        QMessageBox.information(self, "処理完了", message)

        # モデルが生成された場合、ビューワーにロード
        if os.path.exists(os.path.join(self.output_dir, "mesh.obj")):
            self.model_viewer.load_model(os.path.join(self.output_dir, "mesh.obj"))
            self.tabs.setCurrentWidget(self.model_viewer)

    def update_ui_state(self):
        """UIの状態を更新"""
        # 画像と出力フォルダの選択状態に応じてボタンの有効/無効を設定
        buttons_enabled = self.image_dir is not None and self.output_dir is not None

        self.align_photos_btn.setEnabled(buttons_enabled)
        self.build_dense_cloud_btn.setEnabled(buttons_enabled)
        self.build_model_btn.setEnabled(buttons_enabled)

    def set_ui_enabled(self, enabled):
        """UI要素の有効/無効を切り替え"""
        self.load_images_btn.setEnabled(enabled)
        self.select_output_btn.setEnabled(enabled)
        self.align_photos_btn.setEnabled(enabled and self.image_dir is not None and self.output_dir is not None)
        self.build_dense_cloud_btn.setEnabled(enabled and self.image_dir is not None and self.output_dir is not None)
        self.build_model_btn.setEnabled(enabled and self.image_dir is not None and self.output_dir is not None)
        self.quality_combo.setEnabled(enabled)
        self.img_size_combo.setEnabled(enabled)
        self.thread_combo.setEnabled(enabled)

    def export_model(self):
        """モデルをエクスポート"""
        if self.tabs.currentWidget() == self.model_viewer and self.model_viewer.loaded_model_path:
            self.model_viewer.export_model()
        else:
            QMessageBox.warning(self, "警告", "エクスポートするモデルがありません。まずモデルを生成または読み込んでください。")

    def show_manual(self):
        """マニュアルを表示"""
        # 実際の実装ではヘルプファイルを開くなどの処理を行う
        QMessageBox.information(self, "マニュアル", "マニュアルは別途ドキュメントを参照してください。")

    def show_about(self):
        """バージョン情報を表示"""
        QMessageBox.about(self, f"バージョン情報 - {APP_NAME}",
                        f"<h2>{APP_NAME} v{APP_VERSION}</h2>"
                        "<p>フォトグラメトリー技術を用いた3Dモデル生成アプリケーション</p>"
                        "<p>画像から3Dモデルを作成するための簡単で効率的なツールです。</p>"
                        "<p>Copyright © 2025</p>")

    def closeEvent(self, event):
        """アプリケーション終了時の処理"""
        # 処理中の場合は確認ダイアログを表示
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(self, '確認',
                                        "処理実行中です。終了しますか？",
                                        QMessageBox.Yes | QMessageBox.No,
                                        QMessageBox.No)

            if reply == QMessageBox.Yes:
                # 処理をキャンセルして終了
                self.processing_thread.terminate()
                self.processing_thread.wait()
                self.save_settings()
                event.accept()
            else:
                event.ignore()
        else:
            # 設定を保存して終了
            self.save_settings()
            event.accept()


def main():
    """メイン関数"""
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    # スタイルシートの設定（オプション）
    try:
        # カスタムスタイルシートを読み込む場合
        # with open("style.qss", "r") as stylesheet:
        #     app.setStyleSheet(stylesheet.read())
        pass
    except FileNotFoundError:
        # スタイルシートが見つからない場合は何もしない
        pass

    # デスクトップの中央に表示
    window = MainWindow()
    window.show()

    # アプリケーションアイコンの設定（オプション）
    try:
        app_icon = QIcon("app_icon.png")  # アイコンファイルのパス
        app.setWindowIcon(app_icon)
    except Exception as e:
        logger.warning(f"アプリケーションアイコンの読み込みに失敗: {e}")

    sys.exit(app.exec_())


if __name__ == "__main__":
    # アプリケーション実行前のシステム要件チェック
    def check_system_requirements():
        """システム要件を確認する関数"""
        # 必要なライブラリのバージョンチェック
        try:
            from PyQt5.QtWidgets import QApplication

            # 最小バージョン要件の定義
            min_versions = {
                'numpy': (1, 19, 0),
                'opencv-python': (4, 5, 0),
                'open3d': (0, 16, 0),
                'pycolmap': (1, 0, 0),
                'PyQt5': (5, 15, 0)
            }

            # バージョンチェック関数
            def check_version(module, version_tuple):
                current_version = tuple(map(int, module.__version__.split('.')[:3]))
                return current_version >= version_tuple

            # 各モジュールのバージョンをチェック
            for module_name, min_version in min_versions.items():
                module = sys.modules[module_name] if module_name in sys.modules else __import__(module_name)
                if not check_version(module, min_version):
                    print(f"警告: {module_name}のバージョンが古すぎます。")
                    return False

            # システムリソースチェック
            total_memory = psutil.virtual_memory().total / (1024 ** 3)  # GB単位
            cpu_count = os.cpu_count()  # Ensure cpu_count is defined here

            if total_memory < 8:
                print("警告: システムのメモリが不足しています（最小8GB推奨）")
                return False

            if cpu_count < 4:
                print("警告: CPUコア数が少なすぎます（最小4コア推奨）")
                return False

            return True

        except ImportError as e:
            print(f"必要なライブラリが見つかりません: {e}")
            return False
        except Exception as e:
            print(f"システム要件チェック中にエラーが発生: {e}")
            return False

    # システム要件チェック
    if check_system_requirements():
        main()
    else:
        print("システム要件を満たしていないため、アプリケーションを起動できません。")
        sys.exit(1)
