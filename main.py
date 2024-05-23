import tkinter
from tkinter import Tk, Label, Button, Entry, StringVar, Frame, Menu

class DatasetExplorerGUI(Tk):
    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        self.title("Dataset Explorer")
        self.geometry("800x400")

        self.photo_gallery = PhotoGallery()

        self.menu_bar = self._build_menu_bar()
        self.config(menu=self.menu_bar)

    def _build_menu_bar(self):
        menu_bar = Menu(self)

        file_menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        return menu_bar

class PhotoGallery(Frame):
    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        pass

if __name__ == "__main__":
    window = DatasetExplorerGUI()
    window.mainloop()