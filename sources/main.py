import re
import time
import argparse
from k_means import KMeans
from k_means_plus import KMeansPlus
from vq_lbg import VQ_LBG
from utils import *


if __name__ == '__main__':
    # Use argparse to load arguments from command line.
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["KM", "KMPP", "VQ_LBG", "ALL"], help="mode of running")
    parser.add_argument("--visualize", type=bool, default=True, help="visualize the result")
    parser.add_argument("--cluster_num", type=int, default=4, help="the number of clusters")
    parser.add_argument("--sample_num", type=int, default=100, help="the number of samples per cluster")
    parser.add_argument("--center", type=str, default="2,2;8,2;5,8;4,4",
                        help="coordinates of the centers of each cluster")
    parser.add_argument("--save_name", type=str, default=None,
                        help="The filename of the generated data, value 'None' will drop the data after running")
    parser.add_argument("--use_saved_data", type=str, default=None,
                        help="Use saved data to ensure the fairness of each test")
    args = parser.parse_args()

    print(toRed("1. Parsing arguments..."))
    print(toBlue("Mode:{}".format(args.mode)))
    print("Visualize:{0}, Use saved data:{1}\n"
          "Cluster_num:{2}, Sample_num:{3}, Center:{4}".format(
          args.visualize, args.use_saved_data, args.cluster_num, args.sample_num, args.center))

    # Generate or load saved data
    if args.use_saved_data is None:
        print(toRed("2. Generating data..."))
        if args.center == 'auto':
            center = np.random.rand(args.cluster_num, 2) * 10
        else:
            center = re.split(',|;', args.center)
            center = np.array([float(x) for x in center]).reshape(-1, 2)
        data = []
        for i in range(args.cluster_num):
            data.append(np.random.randn(args.sample_num, 2) + center[i])
        data = np.concatenate([data[i] for i in range(args.cluster_num)], axis=0)
        print(toGreen("Done"), "\nThe data shape: {}".format(data.shape))
        if args.save_name is not None:
            save_path = "../data/" + args.save_name + '.npy'
            np.save(save_path, data)
            print("The generated data has been stored in {}".format(save_path))
    else:
        print(toRed("2. Loading saved data..."))
        file_path = "../data/" + args.use_saved_data + '.npy'
        data = np.load(file_path)
        print("Done. The data shape: {}".format(data.shape))

    # Process data using the chosen algorithm(s)
    print(toRed("3. Starting clustering..."))
    if args.mode == "KM":
        kmeans = KMeans(args.cluster_num, data.shape[0], max_iter=100)
        labels, centers, it = kmeans.fit(data)
        print(toGreen('Done'), '\nPredicted labels:\n', labels, '\nPredicted cluster center points:\n', centers)
        print(toRed("4. Evaluating..."))
        score = evaluate(data, labels)
        print("score:{}".format(score))
        if args.visualize == True:
            print(toRed("5. Visualizing..."))
            print(toGreen("Done. Check the figure on screen, close it to exit the program."))
            visualize_single(args.cluster_num, data, labels, centers)
    elif args.mode == "KMPP":
        kmeansplus = KMeansPlus(args.cluster_num, data.shape[0], max_iter=100)
        labels, centers = kmeansplus.fit(data)
        print(toGreen('Done'), '\nPredicted labels:\n', labels, '\nPredicted cluster center points:\n', centers)
        print(toRed("4. Evaluating..."))
        score = evaluate(data, labels)
        print("score:{}".format(score))
        if args.visualize == True:
            print(toRed("5. Visualizing..."))
            print(toGreen("Done. Check the figure on screen, close it to exit the program."))
            visualize_single(args.cluster_num, data, labels, centers)
    elif args.mode == "VQ_LBG":
        vqlbg = VQ_LBG(args.cluster_num, data.shape[0], max_iter=300, epsilon=0.01)
        labels, centers= vqlbg.fit(data)
        print(toGreen('Done'), '\nPredicted labels:\n', labels, '\nPredicted cluster center points:\n', centers)
        print(toRed("4. Evaluating..."))
        score = evaluate(data, labels)
        print("score:{}".format(score))
        if args.visualize == True:
            print(toRed("5. Visualizing..."))
            print(toGreen("Done. Check the figure on screen, close it to exit the program."))
            visualize_single(args.cluster_num, data, labels, centers)
    elif args.mode == "ALL":
        kmeans = KMeans(args.cluster_num, data.shape[0], max_iter=300)
        kmeansplus = KMeansPlus(args.cluster_num, data.shape[0], max_iter=300)
        vqlbg = VQ_LBG(args.cluster_num, data.shape[0], max_iter=1000, epsilon=0.01)
        time_start = time.time()
        labels_1, centers_1 = kmeans.fit(data)
        time_1 = time.time()
        labels_2, centers_2 = kmeansplus.fit(data)
        time_2 = time.time()
        labels_3, centers_3 = vqlbg.fit(data)
        time_3 = time.time()
        labels = [labels_1, labels_2, labels_3]
        centers = [centers_1, centers_2, centers_3]
        time_all = [time_1-time_start, time_2-time_1, time_3-time_2]
        print(toGreen("Done"))
        print(toRed("4. Evaluating..."))
        score = evaluate(data, labels)
        print("score of k-means:{0}\nscore of k-means++:{1}\nscore of vq_lbg:{2}".format(score[0], score[1], score[2]))
        if args.visualize == True:
            print(toRed("5. Visualizing..."))
            print(toGreen("Done. Check the figure on screen, close it to exit the program."))
            visualize_multi(args.cluster_num, data, labels, centers, score, time_all)

