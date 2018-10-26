package net.moli9ma.deeplearning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;

public class MnistLoader {

    public static File MNIST = new File("dataset/mnist");
    public static File IMAGES = new File("output/images");

    public static File WEIGHTS = new File("output/weights");

    public static File TrainImages = new File(MNIST, "train-images.idx3-ubyte");
    public static File TrainLabels = new File(MNIST, "train-labels.idx1-ubyte");
    public static File TestImages = new File(MNIST, "t10k-images.idx3-ubyte");
    public static File TestLabels = new File(MNIST, "t10k-labels.idx1-ubyte");
    public static File SampleWeights = new File(MNIST, "sample_weight.txt");


    public static File TrainImagesOutput = new File(IMAGES, "train");
    public static File TestImagesOutput = new File(IMAGES, "test");
    public static File SampleImagesOutput = new File(IMAGES, "sample");
    public static File OptimizerImages = new File(IMAGES, "optimizer");
    public static File WeightImages = new File(IMAGES, "weight");


    /**
     * イメージの数です。
     */
    public final int size;
    /**
     * １イメージあたりの行数です。
     */
    public final int rows;
    /**
     * １イメージあたりの列数です。
     */
    public final int columns;
    /**
     * イメージあたりのピクセル数です。(rows * columns)
     */
    public final int imageSize;
    /**
     * イメージ(複数)です。
     * 各イメージはフラット化して格納します。
     */
    private final byte[][] images;
    /**
     * ラベル(複数)です。
     * 各要素には0x00から0x09の値が格納されます。
     */
    private final byte[] labels;


    public MnistLoader(File image, File label) throws IOException {
        try (DataInputStream in = new DataInputStream(new FileInputStream(image))) {
            int header = in.readInt();
            if (header != 0x00000803)
                throw new IOException("Invalid image header");
            size = in.readInt();
            rows = in.readInt();
            columns = in.readInt();
            imageSize = rows * columns;
            images = new byte[size][imageSize];
            for (int i = 0; i < size; ++i)
                in.readFully(images[i]);
        }
        try (DataInputStream in = new DataInputStream(new FileInputStream(label))) {
            int header = in.readInt();
            if (header != 0x00000801)
                throw new IOException("Invalid label header");
            if (in.readInt() != size)
                throw new IOException("Invalid label size");
            labels = new byte[size];
            in.readFully(labels);
        }
    }

    public INDArray normalizedImages() {
        INDArray norm = Nd4j.create(size, imageSize);
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < imageSize; ++j)
                norm.putScalar(i, j, (images[i][j] & 0xff) / 255.0);
        return norm;
    }

    public INDArray oneHotLabels() {
        INDArray oneHot = Nd4j.create(size, 10);
        for (int i = 0; i < size; ++i)
            oneHot.putScalar(i, labels[i], 1);
        return oneHot;
    }

    /**
     * 指定された位置のラベル値を返します。
     * @param index 取得する位置を指定します。
     * @return 0から9の範囲でラベル値を返します。
     */
    public int label(int index) {
        return labels[index];
    }

    /**
     * 指定したインデックスのイメージをPNGファイルとして出力します。
     *
     * @param index 出力するイメージのインデックスを指定します。
     * @param path 出力先のファイルを指定します。
     * @throws IOException
     */
    public void writePngFile(int index, File path) throws IOException {
        BufferedImage image = new BufferedImage(columns, rows, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = image.createGraphics();
        try (Closeable cl = () -> g.dispose()) {
            byte[] imageBytes = images[index];
            for (int i = 0, r = 0; r < rows; ++r) {
                for (int c = 0; c < columns; ++c) {
                    int b = imageBytes[i++] & 0xff;
                    g.setColor(new Color(b, b, b));
                    g.fillRect(c, r, 1, 1);
                }
            }
        }
        ImageIO.write(image, "png", path);
    }
}
