import wx

if __name__ == "__main__":
    app = wx.App()
    frame = wx.Frame(None, -1, "Hello World")
    frame.Show()
    app.MainLoop()