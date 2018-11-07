package net.moli9ma.deeplearning.c5;

import net.moli9ma.deeplearning.MnistLoader;
import net.moli9ma.deeplearning.NdUtil;
import net.moli9ma.deeplearning.TwoLayerNetBackPropagation;
import net.moli9ma.deeplearning.TwolayerNetParameter;
import net.moli9ma.deeplearning.layer.AddLayer;
import net.moli9ma.deeplearning.layer.AddLayerData;
import net.moli9ma.deeplearning.layer.MulLayer;
import net.moli9ma.deeplearning.layer.MulLayerData;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class BuyAppleOrange {

    @Test
    void t1() {

        double apple = 100;
        double appleNum = 2;
        double orange = 150;
        double orangeNum = 3;
        double tax = 1.1;

// layer
        MulLayer mulAppleLayer = new MulLayer();
        MulLayer mulOrangeLayer = new MulLayer();
        AddLayer addAppleOrangeLayer = new AddLayer();
        MulLayer mulTaxLayer = new MulLayer();

        // forward
        double applePrice = mulAppleLayer.forward(new MulLayerData(apple, appleNum));
        double orangePrice = mulOrangeLayer.forward(new MulLayerData(orange, orangeNum));

        double allPrice = addAppleOrangeLayer.foraward(new AddLayerData(applePrice, orangePrice));
        double price = mulTaxLayer.forward(new MulLayerData(allPrice, tax));

        // backward
        double dprice = 1;
        MulLayerData dMulTaxLayerData = mulTaxLayer.backward(dprice);
        double dallPrice = dMulTaxLayerData.x;
        double dTax = dMulTaxLayerData.y;

        AddLayerData data = addAppleOrangeLayer.backward(dallPrice);
        double dApplePrice = data.x;
        double dOrangePrice = data.y;

        MulLayerData MulOrangeLayerData = mulOrangeLayer.backward(dOrangePrice);
        double dOrange = MulOrangeLayerData.x;
        double dOrangeNum = MulOrangeLayerData.y;

        MulLayerData MulAppleLayerData = mulAppleLayer.backward(dApplePrice);
        double dApple = MulAppleLayerData.x;
        double dAppleNum = MulAppleLayerData.y;

        assertEquals(715, (int) price);

        assertEquals(2.2, dApple);
        assertEquals(110, (int) dAppleNum);

        assertEquals(3.3000000000000003, dOrange);
        assertEquals(165, (int) dOrangeNum);

        assertEquals(650, (int) dTax);
    }

}
