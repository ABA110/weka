/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka;

import java.awt.BorderLayout;
import java.util.Random;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.PrincipalComponents;
import weka.attributeSelection.Ranker;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.filters.supervised.instance.SMOTE;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

/**
 *
 * @author A.Bagheri
 */
public class FeatureSelectedClassifierGainR {

    public static void main(String args[]) throws Exception {

        ConverterUtils.DataSource source = new ConverterUtils.DataSource("C:/Users/A.Bagheri/Desktop/Training Dataset.arff");
        Instances dataset = source.getDataSet();

        dataset.setClassIndex(dataset.numAttributes() - 1);

        System.out.println("SMOTE and Training in 10 Fold Cross validation in Process:");
        // System.out.println(dataset);

        FilteredClassifier fc = new FilteredClassifier();
        SMOTE smote = new SMOTE();
        smote.setPercentage(26);
        RandomForest RF = new RandomForest();
        RF.setNumFeatures(5);
        RF.setNumTrees(100);

		  // Print header and instances.
        fc.setFilter(smote);
        fc.setClassifier(RF);
        
        AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
        GainRatioAttributeEval eval = new GainRatioAttributeEval();
        Ranker search = new Ranker();
        search.setNumToSelect(5);
        
        classifier.setClassifier(fc);
        classifier.setEvaluator(eval);
        classifier.setSearch(search);
        // 10-fold cross-validation
        Evaluation evaluation = new Evaluation(dataset);
        evaluation.crossValidateModel(classifier, dataset, 10, new Random(1));
        System.out.println(evaluation.toSummaryString());
        

        System.out.println(evaluation.toSummaryString("Results Test:\n", false));
        System.out.println(evaluation.toMatrixString());

        ///////////////////////////////////////
        ThresholdCurve tc = new ThresholdCurve();
        int classIndex = 0;
        Instances result = tc.getCurve(evaluation.predictions(), classIndex);

        // plot curve
        ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
        vmc.setROCString("(Area under ROC = "
                + Utils.doubleToString(tc.getROCArea(result), 4) + ")");
        vmc.setName(result.relationName());
        PlotData2D tempd = new PlotData2D(result);
        tempd.setPlotName(result.relationName());
        tempd.addInstanceNumberAttribute();
        // specify which points are connected
        boolean[] cp = new boolean[result.numInstances()];
        for (int n = 1; n < cp.length; n++) {
            cp[n] = true;
        }
        tempd.setConnectPoints(cp);
        // add plot
        vmc.addPlot(tempd);

        // display curve
        String plotName = vmc.getName();
        final javax.swing.JFrame jf
                = new javax.swing.JFrame("Weka Classifier Visualize: " + plotName);
        jf.setSize(500, 400);
        jf.getContentPane().setLayout(new BorderLayout());
        jf.getContentPane().add(vmc, BorderLayout.CENTER);
        jf.addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent e) {
                jf.dispose();
            }
        });
        jf.setVisible(true);

    }

}
