import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QLabel, QFileDialog, QMessageBox, 
                             QProgressBar, QMainWindow, QCheckBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from tensorflow.keras.models import load_model
import time

class PredictionThread(QThread):
    """预测线程，避免UI卡顿"""
    prediction_done = pyqtSignal(str, float, np.ndarray, float)  # 结果、置信度、热图、原始预测值

    def __init__(self, model, image_path, invert_prediction=False):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.invert_prediction = invert_prediction  # 是否反转预测结果
        
        # 从模型获取输入尺寸
        if model is not None and model.input_shape is not None:
            self.input_shape = (model.input_shape[1], model.input_shape[2])
        else:
            self.input_shape = (224, 224)  # 默认尺寸

    def run(self):
        try:
            # 图像预处理
            img = cv2.imread(self.image_path)
            if img is None:
                raise ValueError("无法读取图像文件")
                
            print(f"模型期望输入尺寸: {self.input_shape}")

            # 调整大小为模型期望的尺寸
            img_resized = cv2.resize(img, self.input_shape)

            # 转为RGB（与训练时图像通道顺序一致）
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            # 注意：模型已经内置了预处理层，不需要额外归一化
            # 扩展维度以匹配模型输入
            img_input = np.expand_dims(img_rgb, axis=0)

            # 模型预测
            start_time = time.time()
            prediction = self.model.predict(img_input)
            end_time = time.time()

            # 获取原始预测值
            pred_value = prediction[0][0]
            print(f"原始预测值: {pred_value}")

            # 确定预测类别和置信度
            if self.invert_prediction:
                # 反转逻辑：pred_value 视为"正常"概率
                class_label = "正常" if pred_value > 0.5 else "肺炎"
                confidence = pred_value if class_label == "正常" else (1 - pred_value)
                print(f"反转预测: 预测为{class_label}，置信度={confidence:.4f}")
            else:
                # 原始逻辑：pred_value 视为"肺炎"概率
                class_label = "肺炎" if pred_value > 0.5 else "正常"
                confidence = pred_value if class_label == "肺炎" else (1 - pred_value)
                print(f"原始预测: 预测为{class_label}，置信度={confidence:.4f}")

            # 模拟热图（简单示例，实际可结合Grad-CAM等生成）
            heatmap = cv2.applyColorMap(cv2.resize(img_resized, (300, 300)), cv2.COLORMAP_JET)

            self.prediction_done.emit(class_label, confidence, heatmap, pred_value)

        except Exception as e:
            print(f"预测错误: {str(e)}")
            empty_heatmap = np.zeros((300, 300, 3), dtype=np.uint8)
            self.prediction_done.emit(f"错误: {str(e)}", 0, empty_heatmap, -1)

class PneumoniaDiagnosisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.input_shape = (224, 224)  # 默认输入尺寸
        self.invert_prediction = False  # 默认不反转
        self.initUI()
        self.loadModel()

    def initUI(self):
        self.setWindowTitle('肺炎X光图像诊断系统')
        self.setGeometry(100, 100, 800, 600)

        main_layout = QVBoxLayout()

        # 标题
        title_label = QLabel('肺炎X光图像诊断系统')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title_label)

        # 模型状态
        self.model_status = QLabel('模型状态: 未加载')
        main_layout.addWidget(self.model_status)
        
        # 模型输入尺寸
        self.input_size_label = QLabel('模型输入尺寸: 未知')
        self.input_size_label.setStyleSheet("font-size: 12px; color: gray;")
        main_layout.addWidget(self.input_size_label)

        # 反转预测选项
        options_layout = QHBoxLayout()
        self.invert_checkbox = QCheckBox('反转预测结果 (若模型输出异常)')
        self.invert_checkbox.stateChanged.connect(self.onInvertCheckboxChanged)
        options_layout.addWidget(self.invert_checkbox)
        main_layout.addLayout(options_layout)

        # 图像显示区域（原始图 + 结果&热图）
        display_layout = QHBoxLayout()

        # 左侧原始图像
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.image_label = QLabel('请上传X光图像')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background-color: #f5f5f5;")

        left_layout.addWidget(QLabel('原始图像:'))
        left_layout.addWidget(self.image_label)

        # 右侧结果和热图
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.result_label = QLabel('预测结果: 未进行预测')
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px 0;")

        self.confidence_label = QLabel('置信度: 0%')
        self.confidence_label.setStyleSheet("font-size: 14px; margin: 5px 0;")

        self.raw_pred_label = QLabel('原始预测值: N/A')
        self.raw_pred_label.setStyleSheet("font-size: 12px; color: blue; margin: 5px 0;")

        self.invert_status_label = QLabel('预测逻辑: 原始 (肺炎>0.5)')
        self.invert_status_label.setStyleSheet("font-size: 12px; color: purple; margin: 5px 0;")

        self.heatmap_label = QLabel('预测热图')
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setMinimumSize(300, 300)
        self.heatmap_label.setStyleSheet("border: 1px solid #ccc; background-color: #f5f5f5;")

        right_layout.addWidget(self.result_label)
        right_layout.addWidget(self.confidence_label)
        right_layout.addWidget(self.raw_pred_label)
        right_layout.addWidget(self.invert_status_label)
        right_layout.addWidget(QLabel('预测热图:'))
        right_layout.addWidget(self.heatmap_label)

        display_layout.addWidget(left_panel)
        display_layout.addWidget(right_panel)
        main_layout.addLayout(display_layout)

        # 操作按钮
        button_layout = QHBoxLayout()

        self.upload_button = QPushButton('上传X光图像')
        self.upload_button.setMinimumHeight(40)
        self.upload_button.clicked.connect(self.uploadImage)

        self.predict_button = QPushButton('开始诊断')
        self.predict_button.setMinimumHeight(40)
        self.predict_button.clicked.connect(self.predict)
        self.predict_button.setEnabled(False)  # 未上传图时禁用

        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.predict_button)
        main_layout.addLayout(button_layout)

        # 状态栏
        self.statusBar().showMessage('就绪')

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.current_image_path = None
        self.prediction_thread = None

    def onInvertCheckboxChanged(self, state):
        self.invert_prediction = (state == Qt.Checked)
        if self.invert_prediction:
            self.invert_status_label.setText('预测逻辑: 已反转 (正常>0.5)')
            self.invert_status_label.setStyleSheet("font-size: 12px; color: purple; font-weight: bold; margin: 5px 0;")
        else:
            self.invert_status_label.setText('预测逻辑: 原始 (肺炎>0.5)')
            self.invert_status_label.setStyleSheet("font-size: 12px; color: purple; margin: 5px 0;")

        QMessageBox.information(self, '提示', 
                                '预测逻辑已' + ('反转' if self.invert_prediction else '恢复') + 
                                '，新设置将在下一次预测时生效。\n\n' +
                                '注意: 反转预测是临时解决方案，建议尽快重新训练模型。')

    def loadModel(self):
        try:
            # 模型路径，根据实际训练好的模型位置调整
            model_path = r"C:\Users\86191\code\optimized_pneumonia_model.h5"  
            if not os.path.exists(model_path):
                QMessageBox.warning(self, '警告', f'模型文件不存在: {model_path}')
                return

            self.statusBar().showMessage('正在加载模型...')
            self.model = load_model(model_path)

            # 获取模型输入尺寸
            if self.model.input_shape is not None and len(self.model.input_shape) >= 4:
                self.input_shape = (self.model.input_shape[1], self.model.input_shape[2])
            
            print("模型加载成功！")
            print(f"模型输入形状: {self.model.input_shape}")
            print(f"模型输出形状: {self.model.output_shape}")
            print(f"所需输入尺寸: {self.input_shape}")

            self.model_status.setText(f'模型状态: 已加载 ({model_path})')
            self.input_size_label.setText(f'模型输入尺寸: {self.input_shape[0]}×{self.input_shape[1]}')
            self.statusBar().showMessage(f'模型加载完成 - 输入尺寸: {self.input_shape[0]}×{self.input_shape[1]}')

        except Exception as e:
            self.model_status.setText('模型状态: 加载失败')
            QMessageBox.critical(self, '错误', f'模型加载失败: {str(e)}')
            self.statusBar().showMessage('模型加载失败')

    def uploadImage(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择X光图像', '', 
            '图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*)'
        )
        if file_path:
            self.current_image_path = file_path
            self.displayImage(file_path)
            self.predict_button.setEnabled(True)
            self.statusBar().showMessage(f'已选择图像: {os.path.basename(file_path)}')

    def displayImage(self, image_path):
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                self.image_label.width(), 
                self.image_label.height(),
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
            self.image_label.setAlignment(Qt.AlignCenter)
        else:
            QMessageBox.warning(self, '警告', '无法加载图像')

    def predict(self):
        if not self.model or not self.current_image_path:
            return

        self.upload_button.setEnabled(False)
        self.predict_button.setEnabled(False)
        self.predict_button.setText('诊断中...')
        self.statusBar().showMessage('正在进行诊断...')

        self.prediction_thread = PredictionThread(self.model, self.current_image_path, self.invert_prediction)
        self.prediction_thread.prediction_done.connect(self.onPredictionDone)
        self.prediction_thread.start()

    def onPredictionDone(self, result, confidence, heatmap, raw_pred):
        self.upload_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        self.predict_button.setText('开始诊断')

        if result.startswith('错误'):
            QMessageBox.critical(self, '错误', result)
            self.statusBar().showMessage('诊断失败')
            return

        self.result_label.setText(f'预测结果: {result}')
        confidence_percent = confidence * 100
        self.confidence_label.setText(f'置信度: {confidence_percent:.2f}%')
        self.raw_pred_label.setText(f'原始预测值: {raw_pred:.6f}')

        if result == "肺炎":
            self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: red; margin: 10px 0;")
        else:
            self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: green; margin: 10px 0;")

        # 显示热图
        height, width, channel = heatmap.shape
        bytes_per_line = 3 * width
        q_img = QImage(heatmap.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(
            self.heatmap_label.width(), 
            self.heatmap_label.height(),
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.heatmap_label.setPixmap(pixmap)
        self.heatmap_label.setAlignment(Qt.AlignCenter)

        invert_note = "（已反转）" if self.invert_prediction else ""
        self.statusBar().showMessage(f'诊断完成: {result} (置信度: {confidence_percent:.2f}%) {invert_note}')

        QMessageBox.information(
            self, '诊断结果', 
            f'预测结果: {result}\n置信度: {confidence_percent:.2f}%\n原始预测值: {raw_pred:.6f}\n预测逻辑: {"已反转" if self.invert_prediction else "原始"}\n\n'
            f'注意: 此结果仅供参考，不能替代专业医疗诊断。'
        )

if __name__ == '__main__':
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    app = QApplication(sys.argv)
    window = PneumoniaDiagnosisApp()
    window.show()
    sys.exit(app.exec_())