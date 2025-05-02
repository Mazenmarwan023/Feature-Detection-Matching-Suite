import sys
from PyQt5.QtWidgets import QApplication
from Harris import Harris


if __name__ == "__main__":
    app=QApplication(sys.argv)
    window=Harris()
    window.show()    
    sys.exit(app.exec_())