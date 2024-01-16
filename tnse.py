from sklearn.manifold import TSNE
import seaborn as sns

def plot_tsne(embeddings, save_dir, save_name):
    tsne = TSNE()
    tsne_result = tsne.fit_transform(embeddings)

    fig = sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1])
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    fig.set_xlim(lim)
    fig.set_ylim(lim)
    fig.set_aspect('equal')
    fig.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    fig = fig.get_figure()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, save_name + '_tsne_plot.png'))
    fig.savefig(os.path.join(save_dir, save_name + '_tsne_plot.pdf'))

if __name__ == "__main__":
    embeddings = np.load("embeddings.npy")
    plot_tsne(embeddings, "./", "test")
    
