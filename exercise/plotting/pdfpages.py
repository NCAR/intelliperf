import sys
import numpy as np

try:
    from  matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf = PdfPages('blank.pdf')
except Exception as e:
    print ('ERROR: matplotlib module is not loaded: %s'%str(e))
    sys.exit(-1)

TITLE_SIZE = 20
SUBTITLE_SIZE = 16
TEXT_SIZE = 14
LABEL_SIZE = 16
LINEWIDTH = 3

def main():

    num_pages = 3
    num_points = 100
    y_range = 10
    num_yticks = 5

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.linspace(0, 1, num=num_points, endpoint=True)

    for idx in range(num_pages):

        y = np.random.random(num_points)*y_range

        ax.scatter(x, y, alpha=0.4, color='b', linewidth=LINEWIDTH)

        ax.set_title("Title %d"%idx, fontsize=TITLE_SIZE)
        yticks = np.linspace(0, y_range, num=num_yticks, endpoint=True)
        ytick_labels = ['{0:5.3f}'.format(ytick) for ytick in yticks]
        ax.set_yticks(yticks, ytick_labels)
        ax.set_xlabel("Samples", fontsize=LABEL_SIZE)
        ax.set_ylabel("Random value", fontsize=LABEL_SIZE)

        pdf.savefig()
        plt.cla()

    pdf.close()

if __name__ == "__main__":
    main()
