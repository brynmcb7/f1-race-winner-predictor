import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from f1_project.cli import main
if __name__ == "__main__":
    main()