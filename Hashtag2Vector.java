/**
 * This project is developed by the Intelligent Information Processing Lab, Nankai University, Tianjin, China. (http://dm.nankai.edu.cn/)
 * It follows the GPLv3 license. Feel free to use it for non-commercial purpose and please cite our paper:
 * @inproceedings{Hashtag2Vec,
 *   author    = {Jie Liu and Zhicheng He and Yalou Huang},
 *   title     = {Hashtag2Vec: Learning Hashtag Representation with Relational Hierarchical Embedding Model},
 *   booktitle = {Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, {IJCAI} 2018, July 13-19, 2018, Stockholm, Sweden.},
 *   pages     = {3456--3462},
 *   year      = {2018},
 *   doi       = {10.24963/ijcai.2018/480},
 *   }
 * Contact: jliu@nankai.edu.cn, hezhicheng@mail.nankai.edu.cn
*/
package cn.edu.nk.iiplab.hzc.hashtag2vec;

import cn.edu.nk.iiplab.hzc.basic.MF;
import cn.edu.nk.iiplab.hzc.basic.matrix.DenseMatrix;
import cn.edu.nk.iiplab.hzc.basic.matrix.RowSparseMatrix;
import cn.edu.nk.iiplab.hzc.basic.thread.MultiThread;

public class Hashtag2Vector extends MF {
    // Input matrices
    public RowSparseMatrix M_h_h; // hashtag-hashtag matrix
    public RowSparseMatrix M_h_t; // hashtag-tweet matrix
    public RowSparseMatrix M_t_w; // tweet-word matrix
    public RowSparseMatrix M_w_w; // word-word matrix
    // Output factor matrices
    public DenseMatrix W_t;
    public DenseMatrix W_w;
    public DenseMatrix W_h;
    // Hyper parameters
    public double alpha = 1; // Weight of the hashtag-hashtag factorization loss
    public double beta = 1; // Weight of the hashtag-tweet factorization loss
    public double gamma = 1; // Weight of the tweet-word factorization loss
    public double theta = 1; // Weight of the word-word factorization loss
    public double lambda = 2; // Weight of the L2 regularization terms
    public int latent_dim = 20; // Latent dimension of the factor matrices
    public int iter_num = 500; // Maximum iteration number
    public int neg_samp_num = 30; // Number of negative samples

    public int index = 0;
    // File paths
    public String data_path = ""; // Data folder path
    public String log_path = "";  // Running log

    public void initialize(String inDir, String h_h_name, String h_t_name, String t_w_name, String w_w_name, String log_path) throws Exception {
        this.data_path = inDir;
        this.log_path = log_path;

        // Load input matrices
        M_h_h = new RowSparseMatrix();
        M_h_h.loadMy(inDir + h_h_name);

        M_h_t = new RowSparseMatrix();
        M_h_t.loadMy(inDir + h_t_name);

        M_t_w = new RowSparseMatrix();
        M_t_w.loadMy(inDir + t_w_name);

        M_w_w = new RowSparseMatrix();
        M_w_w.loadMy(inDir + w_w_name);

        // Randomly initialize the factor matrices
        W_t = new DenseMatrix();
        W_t.randInit(latent_dim, M_h_t.iNumOfColumn);
        W_w = new DenseMatrix();
        W_w.randInit(latent_dim, M_w_w.iNumOfRow);
        W_h = new DenseMatrix();
        W_h = matrixMultiply(M_h_t, W_t.transpose()).transpose();
    }

    public void optimize() throws Exception {
        // Multiplicative update iterations
        int step = 0;
        long start = System.currentTimeMillis();
        while (step < iter_num) {
            System.out.println("Start iteration " + step + ":");
            System.out.println("Update W_t with the multiplicative update rules,");
            // Multiplicative update rules for W_t
            // Numerator for the multiplicative update
            RowSparseMatrix devStw_Mtw = matrixNonlinearDfWeightedMultiply(W_t.transpose(), W_w, M_t_w);
            RowSparseMatrix devSht_Mht = matrixNonlinearDfWeightedMultiply(W_h.transpose(), W_t, M_h_t);
            RowSparseMatrix Nolinear_tw = matrixNonlinearFDfWeightedMultiply(W_t.transpose(), W_w, M_t_w.binaryWeight(neg_samp_num));
            RowSparseMatrix Nolinear_ht = matrixNonlinearFDfWeightedMultiply(W_h.transpose(), W_t, M_h_t.binaryWeight(neg_samp_num));
            DenseMatrix Nmrt_t = matrixMultiplyNumber(matrixMultiply(devStw_Mtw, W_w.transpose()).transpose(), gamma);
            Nmrt_t = matrixAdd(Nmrt_t, matrixMultiplyNumber(matrixMultiply(W_h, devSht_Mht), beta));
            // Denominator for the multiplicative update
            DenseMatrix Dnmnt_t = matrixMultiplyNumber(matrixMultiply(Nolinear_tw, W_w.transpose()).transpose(), gamma);
            Dnmnt_t = matrixAdd(Dnmnt_t, matrixMultiplyNumber(matrixMultiply(W_h, Nolinear_ht), beta));
            // L2 regularization
            Dnmnt_t = matrixAdd(Dnmnt_t, matrixMultiplyNumber(W_t, lambda));
            //Update W_t here
            multiplicativeUpdate(Nmrt_t, Dnmnt_t, W_t);

            System.out.println("Update W_h with the multiplicative update rules,");
            // Multiplicative update rules for W_h
            // Numerator for the multiplicative update
            devSht_Mht = matrixNonlinearDfWeightedMultiply(W_h.transpose(), W_t, M_h_t);
            RowSparseMatrix devShh_Mhh = matrixNonlinearDfWeightedMultiply(W_h.transpose(), W_h, M_h_h);
            DenseMatrix Nmrt_h = matrixMultiplyNumber(matrixMultiply(devSht_Mht, W_t.transpose()).transpose(), beta);
            Nmrt_h = matrixAdd(Nmrt_h, matrixMultiplyNumber(matrixMultiply(devShh_Mhh, W_h.transpose()).transpose(), alpha));
            // Denominator for the multiplicative update
            Nolinear_ht = matrixNonlinearFDfWeightedMultiply(W_h.transpose(), W_t, M_h_t.binaryWeight(neg_samp_num));
            DenseMatrix Dnmnt_h = matrixMultiplyNumber(matrixMultiply(Nolinear_ht, W_t.transpose()).transpose(), beta);
            RowSparseMatrix Nolinear_hh = matrixNonlinearFDfWeightedMultiply(W_h.transpose(), W_h, M_h_h.binaryWeight(neg_samp_num));
            Dnmnt_h = matrixAdd(Dnmnt_h, matrixMultiplyNumber(matrixMultiply(Nolinear_hh, W_h.transpose()).transpose(), alpha));
            // L2 regularization
            Dnmnt_h = matrixAdd(Dnmnt_h, matrixMultiplyNumber(W_h, lambda));
            //Update W_h here
            multiplicativeUpdate(Nmrt_h, Dnmnt_h, W_h);

            System.out.println("Update W_w with the multiplicative update rules,");
            // Multiplicative update rules for W_w
            // Numerator for the multiplicative update
            devStw_Mtw = matrixNonlinearDfWeightedMultiply(W_t.transpose(), W_w, M_t_w);
            RowSparseMatrix devSww_Mww = matrixNonlinearDfWeightedMultiply(W_w.transpose(), W_w, M_w_w);
            DenseMatrix Nmrt_w = matrixMultiplyNumber(matrixMultiply(W_t, devStw_Mtw), gamma);
            Nmrt_w = matrixAdd(Nmrt_w, matrixMultiplyNumber(matrixMultiply(devSww_Mww, W_w.transpose()).transpose(), theta));
            // Denominator for the multiplicative update
            Nolinear_tw = matrixNonlinearFDfWeightedMultiply(W_t.transpose(), W_w, M_t_w.binaryWeight(neg_samp_num));
            DenseMatrix Dnmnt_w = matrixMultiplyNumber(matrixMultiply(W_t, Nolinear_tw), gamma);
            RowSparseMatrix Nolinear_ww = matrixNonlinearFDfWeightedMultiply(W_w.transpose(), W_w, M_w_w.binaryWeight(neg_samp_num));
            Dnmnt_w = matrixAdd(Dnmnt_w, matrixMultiplyNumber(matrixMultiply(Nolinear_ww, W_w.transpose()).transpose(), theta));
            // L2 regularization
            Dnmnt_w = matrixAdd(Dnmnt_w, matrixMultiplyNumber(W_w, lambda));
            //Update W_w here
            multiplicativeUpdate(Nmrt_w, Dnmnt_w, W_w);

            System.out.println("End iteration " + step + ".");
            step++;
        }
        long end = System.currentTimeMillis();
        System.out.println("The update took: " + (end - start) / 1000 + "s.");
        String path = "W_h_" + iter_num + "_" + neg_samp_num + "_" + index + ".txt";
        W_h.save(path);

    }

    public void evaluate(String hashtag_cindices, String hashtag_labels, int hashtag_cluster_num,
                         String tweet_cindices, String tweet_labels, int tweet_cluster_num,
                         int[] pmi_top) throws Exception {
        Logger logger = new Logger(log_path, false);
        Performance per = new Performance(data_path, logger.folder_path);
        double[] res = per.evaluate(W_h.transpose(), W_t.transpose(), W_w.transpose(), M_w_w, logger, hashtag_cindices, hashtag_labels, hashtag_cluster_num,
                tweet_cindices, tweet_labels, tweet_cluster_num, pmi_top);
        System.out.println("Hashtag clustering evaluation: ");
        System.out.println("Purity: " + res[0] + ",");
        System.out.println("NMI: " + res[1] + ",");
        System.out.println("H-Score: " + res[2] + ".");

        System.out.println("Tweet clustering evaluation: ");
        System.out.println("Purity: " + res[3] + ",");
        System.out.println("NMI: " + res[4] + ",");
        System.out.println("H-Score: " + res[5] + ".");

        System.out.println("Topic coherence: ");
        System.out.println("Top 5: " + res[6] + ",");
        System.out.println("Top 10: " + res[7] + ",");
        System.out.println("Top 20: " + res[8] + ".");

    }

    public void start_setting(Hashtag2Vector h2vec, int dim, int iter, int neg, int ind){
        h2vec.latent_dim = dim;
        h2vec.iter_num = iter;
        h2vec.neg_samp_num = neg;
        h2vec.index = ind;
        //return h2vec;
    }

    public static void main(String[] args) throws Exception {
        //String data_path = "data\\synthetic\\";
        String data_path = "data\\withoutbert\\";
        String h_hName = "M_h_h.txt";
        String h_tName = "M_h_t.txt";
        String w_wName = "M_w_w.txt";
        String t_wName = "M_t_w.txt";

        String log_path = "Trainset_tweet_hh_component_2015";
        /*
        The Files hashtag_index and tweet_index are just indexes with leading count of used indexes/hashtags
        The label files are the clusternumbers they belong to for validation purpose.
         */
        String hashtag_cindices = "hashtag_index.txt";
        String hashtag_labels = "hashtag_label.txt";
        int hashtag_cluster_num = 61;
        String tweet_cindices = "tweet_index.txt";
        String tweet_labels = "tweet_label.txt";
        int tweet_cluster_num = 16;
        int[] pmi_top = {5, 10, 20};

        MultiThread.iMaxThreads = 80; // Number of parallel threads for acceleration

        Hashtag2Vector h2v = new Hashtag2Vector();
        h2v.start_setting(h2v, 16, 5, 10, 0);
        /*h2v.latent_dim = 16; // Dimension of learned embeddings
        h2v.iter_num = 5; // Maximum iteration number
        h2v.neg_samp_num = 5; // Number of negative samplings */
        h2v.initialize(data_path, h_hName, h_tName, t_wName, w_wName, log_path);
        h2v.optimize();

        Hashtag2Vector h2v2 = new Hashtag2Vector();
        h2v2.start_setting(h2v2, 16, 5, 5, 0);
        h2v2.initialize(data_path, h_hName, h_tName, t_wName, w_wName, log_path);
        h2v2.optimize();

        hashtag_cindices = "hashtag2_index.txt";
        hashtag_labels = "hashtag2_label.txt";
        hashtag_cluster_num = 14;
        Hashtag2Vector h2v3 = new Hashtag2Vector();
        h2v3.start_setting(h2v3, 16, 5, 10, 1);
        h2v3.initialize(data_path, h_hName, h_tName, t_wName, w_wName, log_path);
        h2v3.optimize();

        Hashtag2Vector h2v4 = new Hashtag2Vector();
        h2v4.start_setting(h2v4, 16, 5, 5, 1);
        h2v4.initialize(data_path, h_hName, h_tName, t_wName, w_wName, log_path);
        h2v4.optimize();
        System.out.println("Final evaluations: ");
        h2v3.evaluate(hashtag_cindices, hashtag_labels, hashtag_cluster_num, tweet_cindices, tweet_labels, tweet_cluster_num, pmi_top);

    }
}
