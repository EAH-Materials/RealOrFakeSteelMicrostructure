import torch
import os
import sys
import glob
import csv
import random

# pip install PyQt6

from PyQt6.QtGui import QPixmap, QPainter, QPen
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow, QGridLayout, QApplication, QPushButton, QLabel, QWidget, QCheckBox, QTextEdit, QFrame

REAL_IMGS_PATH = "data/real"
FRACTALGEN_PATH = "data/fractalgen"
STYLEGAN_PATH = "data/stylegan"
CSV_PATH = "evaluate"

class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)

class ScaleBar(QWidget):
    def __init__(self, scale_length=100, physical_length=5, parent=None):
        super().__init__(parent)
        self.scale_length = scale_length
        self.physical_length = physical_length
        self.setMinimumHeight(40)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(Qt.GlobalColor.black, 2)
        painter.setPen(pen)

        margin = 10
        x_start = margin
        y_position = self.height() - margin

        painter.drawLine(x_start, y_position, x_start + self.scale_length, y_position)

        tick_size = 5
        painter.drawLine(x_start, y_position - tick_size, x_start, y_position + tick_size)
        painter.drawLine(x_start + self.scale_length, y_position - tick_size, 
                         x_start + self.scale_length, y_position + tick_size)

        text = f"{self.physical_length} μm"
        painter.drawText(x_start + margin, y_position - tick_size, text)

class MainWindow(QMainWindow):
    def random_real(self,):
        return self.real_images.pop()

    def random_stylegan(self,):
        return self.stylegan_images.pop()

    def random_fractalgen(self,):
        return self.fractalgen_images.pop()
    
    def increase_decisioncounter(self,):
        self.decision_counter += 1
        self.setWindowTitle(f"Real or Fake - Decisions Count: {self.decision_counter}")

    def choice_real(self,):
        self.reset_reasons()
        self.save_guess(self.is_fake == False, True)
        self.show_next_image()

    def choice_fake(self,):
        if not self.is_reason_given():
            return

        self.save_guess(self.is_fake == True, False)
        self.reset_reasons()
        self.show_next_image()

    def is_reason_given(self,):
        reason_given = any(cb.isChecked() for cb in self.checkboxes)
        if not reason_given:
            print("Give a reason.")
            return False
        
        if self.checkbox_do.isChecked() and not self.textbox_do.toPlainText().strip():
            print("Other is checked, write a reason in the text field.")
            return False

        if self.checkbox_no.isChecked() and not self.textbox_no.toPlainText().strip():
            print("Other is checked, write a reason in the text field.")
            return False
        return True

    def save_guess(self, guess_was_correct, guess):
        if self.current_img_path == None:
            return 
        
        csv_file_path = os.path.join(self.csv_path, "guesses.csv")
        file_exists = os.path.isfile(csv_file_path)
        with open(csv_file_path, mode="a", newline="") as csv_file:
            fieldnames = ["file", "model", "guess", "is_fake", "guess_is_correct", "N1", "N2", "NO", "NO (Text)", "D1", "D2", "D3", "D4", "DO", "DO (Text)"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=";")
            if not file_exists:
                writer.writeheader()

            writer.writerow({"file": os.path.split(self.current_img_path)[1], "model": self.model, "guess": guess , "is_fake": self.is_fake, "guess_is_correct": guess_was_correct,
                             "N1": self.checkbox_n1.isChecked(),
                             "N2": self.checkbox_n2.isChecked(),
                             "NO": self.checkbox_no.isChecked(),
                             "NO (Text)": self.textbox_no.toPlainText().replace(";", "##"),
                             "D1": self.checkbox_d1.isChecked(),
                             "D2": self.checkbox_d2.isChecked(),
                             "D3": self.checkbox_d3.isChecked(),
                             "D4": self.checkbox_d4.isChecked(),
                             "DO": self.checkbox_do.isChecked(),
                             "DO (Text)": self.textbox_do.toPlainText().replace(";", "##"),
                             })
            
        self.increase_decisioncounter()

    def reset_reasons(self,):
        for cb in self.checkboxes:
            cb.setChecked(False)

        for tb in self.textboxes:
            tb.setText("")

    def show_next_image(self,):
        self.current_img_path = self.select_image()

        if self.current_img_path == None:
            return

        pixmap = QPixmap(self.current_img_path)
        pixmap = pixmap.scaled(pixmap.width()*2, pixmap.height()*2, transformMode=Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(pixmap)

    def stylegan_available(self,):
        return len(self.stylegan_images) > 0
    
    def fractalgen_available(self):
        return len(self.fractalgen_images) > 0
    
    def real_available(self):
        return len(self.real_images) > 0

    def select_image(self):
        fake_real_percentage = 0 # Get real next
        if self.real_available() == False and self.stylegan_available() == False and self.fractalgen_available() == False:
            print("No more images available!")
            self.model = ""
            return None
        
        if self.real_available() and (self.stylegan_available() or self.fractalgen_available()): 
            fake_real_percentage = torch.rand(1)[0]
        elif self.real_available() == False:
            fake_real_percentage = 1 # Get fake next
            
        if fake_real_percentage >= 0.5:
            self.is_fake = True

            generator_percentage = 0 # Get fractal next
            if self.stylegan_available() and self.fractalgen_available():
                generator_percentage = torch.rand(1)[0]
            elif self.stylegan_available():
                generator_percentage = 1 # Get stylegan next

            if generator_percentage >= 0.5:
                self.model = "StyleGANv2"
                return self.random_stylegan()
            else:
                self.model = "FractalGen"
                return self.random_fractalgen()

        elif self.real_available():
            self.is_fake = False
            self.model = "real"
            return self.random_real()

    def enable_textbox_d(self, checkstate):
        self.adjust_textbox_status(self.textbox_do, checkstate)

    def enable_textbox_n(self, checkstate):
        self.adjust_textbox_status(self.textbox_no, checkstate)

    def adjust_textbox_status(self, textbox, checkstate):
        textbox.setReadOnly(Qt.CheckState.Unchecked == checkstate)
        if Qt.CheckState.Unchecked == checkstate:
            textbox.setText("")
        else:
            textbox.setFocus()

    def __init__(self):
        super().__init__()

        self.csv_path = CSV_PATH
        os.makedirs(self.csv_path, exist_ok=True)

        random.seed(0)
        
        self.real_images = glob.glob(os.path.join(REAL_IMGS_PATH, "**/*.jpg"))
        random.shuffle(self.real_images)
        
        self.stylegan_images = glob.glob(os.path.join(STYLEGAN_PATH, "**/*.png"))
        random.shuffle(self.stylegan_images)

        self.fractalgen_images = glob.glob(os.path.join(FRACTALGEN_PATH, "**/*.png"))
        random.shuffle(self.fractalgen_images)

        self.setup_ui()

        self.checkboxes = [
            self.checkbox_n1,
            self.checkbox_n2,
            self.checkbox_no,
            self.checkbox_d1,
            self.checkbox_d2,
            self.checkbox_d3,
            self.checkbox_d4,
            self.checkbox_do,
        ]

        self.textboxes = [
            self.textbox_no,
            self.textbox_do,
        ]

        self.show_next_image()
        self.show()

    def setup_ui(self):
        layout = QGridLayout()

        self.label = QLabel(self)
        self.pixmap = QPixmap()
        self.label.setPixmap(self.pixmap)

        self.decision_counter = 0

        layout.addWidget(self.label, 0, 0, 1, 3)

        scale_bar = ScaleBar(scale_length=200, physical_length=100)        
        layout.addWidget(scale_bar, 0, 0, 1, 3)

        self.add_buttons(layout, row=2)
        layout.addWidget(QHLine(), 3, 0, 1, 3)

        layout_checkboxes_n = self.add_nondomain_issues()

        layout.addLayout(layout_checkboxes_n, 4, 0, 1, 3)
        layout.addWidget(QHLine(), 5, 0, 1, 3)

        layout_checkboxes_d = self.add_domain_issues()

        layout.addLayout(layout_checkboxes_d, 6, 0, 1, 3)

        self.resize(self.pixmap.width(), self.pixmap.height())
        self.setWindowTitle("Real or Fake")
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def add_buttons(self, layout, row):
        self.button1 = QPushButton("Is Fake", self)
        self.button1.clicked.connect(self.choice_fake)
        self.button2 = QPushButton("Is Real")
        self.button2.clicked.connect(self.choice_real)

        layout.addWidget(self.button1, row, 0)
        layout.addWidget(self.button2, row, 1)

    def add_domain_issues(self):
        layout_checkboxes_d = QGridLayout()

        self.checkbox_d1 = QCheckBox("Grain or Phase", self)
        self.checkbox_d2 = QCheckBox("Etching or Contrast", self)
        self.checkbox_d3 = QCheckBox("Morphological or Structural Deviations", self)
        self.checkbox_d4 = QCheckBox("Noise or Artifacts Inherent to Microscopy", self)
        self.checkbox_do = QCheckBox("Other (free text)", self)
        self.textbox_do = QTextEdit("", self)
        self.textbox_do.setReadOnly(True)
        self.checkbox_do.checkStateChanged.connect(self.enable_textbox_d)

        self.textbox_do.setFixedHeight(28)
        self.textbox_do.setFixedWidth(300)

        label_d = QLabel("Domain specific issue (select all that apply)")
        label_d.setStyleSheet("font-weight: bold")
        layout_checkboxes_d.addWidget(label_d, 1, 0,1,2)
        layout_checkboxes_d.addWidget(self.checkbox_d1, 5, 0, 1, 3)
        layout_checkboxes_d.addWidget(self.checkbox_d2, 6, 0, 1, 3)
        layout_checkboxes_d.addWidget(self.checkbox_d3, 7, 0, 1, 3)
        layout_checkboxes_d.addWidget(self.checkbox_d4, 8, 0, 1, 3)
        layout_checkboxes_d.addWidget(self.checkbox_do, 9, 0, 1, 1)
        layout_checkboxes_d.addWidget(self.textbox_do, 9, 1)
        
        return layout_checkboxes_d

    def add_nondomain_issues(self):
        layout_checkboxes_n = QGridLayout()
        label_n = QLabel("Non-domain specific issue (select all that apply)")
        label_n.setStyleSheet("font-weight: bold")

        self.checkbox_n1 = QCheckBox("Visible Artifacts or Grid-Like Patterns", self)
        self.checkbox_n2 = QCheckBox("Lighting and Contrast", self)
        self.checkbox_no = QCheckBox("Other (free text)", self)
        self.textbox_no = QTextEdit("", self)
        self.textbox_no.setReadOnly(True)
        self.checkbox_no.checkStateChanged.connect(self.enable_textbox_n)

        self.textbox_no.setFixedHeight(28)
        self.textbox_no.setFixedWidth(300)

        layout_checkboxes_n.addWidget(label_n, 1, 0,1,2)
        layout_checkboxes_n.addWidget(self.checkbox_n1, 2, 0, 1, 3)
        layout_checkboxes_n.addWidget(self.checkbox_n2, 3, 0, 1, 3)
        layout_checkboxes_n.addWidget(self.checkbox_no, 4, 0, 1, 1)
        layout_checkboxes_n.addWidget(self.textbox_no, 4, 1)
        return layout_checkboxes_n

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    app.exec()