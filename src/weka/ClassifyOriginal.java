/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka;

import java.awt.BorderLayout;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.J48;
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
public class ClassifyOriginal {
    
    
    public static void main(String args[]) throws Exception {
        
         
            ClassifyOriginal classif = new ClassifyOriginal();
            classif.RandomForrest(40, 2);
        classif.RandomForrest(40, 5);
        classif.RandomForrest(40, 9);
        classif.RandomForrest(100, 2);
        classif.RandomForrest(100, 5);
        classif.RandomForrest(100, 9);
        classif.RandomForrest(200, 5);
        classif.RandomForrest(200, 9);
        classif.RandomForrest(300, 9);
        classif.RandomForrest(400, 9);
        /////////////////////////////////
            classif.Jrip(5);
            classif.Jrip(10);
            classif.Jrip(20);
            classif.Jrip(30);
            classif.Jrip(35);
            classif.Jrip(40);
            classif.Jrip(45);
            classif.Jrip(50);
            classif.Jrip(60);
            classif.Jrip(70);
            /////////////////////////////
            classif.ibk(1);
            classif.ibk(2);
            classif.ibk(3);
            classif.ibk(4);
            classif.ibk(5);
            classif.ibk(6);
            classif.ibk(7);
            classif.ibk(8);
            classif.ibk(9);
            classif.ibk(10);
            ////////////////////////////
            classif.classifyJ48(0.1);
            classif.classifyJ48(0.15);
            classif.classifyJ48(0.2);
            classif.classifyJ48(0.25);
            classif.classifyJ48(0.3);
            classif.classifyJ48(0.35);
            classif.classifyJ48(0.4);
            classif.classifyJ48(0.45);
            classif.classifyJ48(0.5);
            classif.classifyJ48(0.7);
            ///////////////////////////////
        

     
    }
    
    
    
    
    
         public void ibk( int k) throws Exception   {
       ConverterUtils.DataSource source = new ConverterUtils.DataSource("C:/Users/A.Bagheri/Desktop/Training Dataset.arff");
        Instances dataset = source.getDataSet();
        
        dataset.setClassIndex(dataset.numAttributes() - 1);
        
        System.out.println("SMOTE and Training in 10 Fold Cross validation Lazy IBk in Process:");
        // System.out.println(dataset);
        SMOTE smote = new SMOTE();
        smote.setPercentage(26);
        IBk ibk = new IBk();
        String [] options = new String[1];
	 options[0] = "-F";
        ibk.setOptions(options);
        ibk.setKNN(k);
		  // Print header and instances.

        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(smote);
        fc.setClassifier(ibk);

        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(fc, dataset, 10, new Random(1));

        System.out.println(eval.toSummaryString("Results Test:\n", false));
        System.out.println(eval.toMatrixString());
        
        ///////////////////////////////////////
        ThresholdCurve tc = new ThresholdCurve();
        int classIndex = 0;
        Instances result = tc.getCurve(eval.predictions(), classIndex);

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
          public void Jrip(int run ) throws Exception   {
       ConverterUtils.DataSource source = new ConverterUtils.DataSource("C:/Users/A.Bagheri/Desktop/Training Dataset.arff");
        Instances dataset = source.getDataSet();
        
        dataset.setClassIndex(dataset.numAttributes() - 1);
        
        System.out.println("SMOTE and Training in 10 Fold Cross validation J48 in Process:");
        // System.out.println(dataset);
        SMOTE smote = new SMOTE();
        smote.setPercentage(26);
        JRip jrip = new JRip();
        jrip.setMinNo(5);
        jrip.setOptimizations(run);
		  // Print header and instances.

        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(smote);
        fc.setClassifier(jrip);

        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(fc, dataset, 10, new Random(1));

        System.out.println(eval.toSummaryString("Results Test:\n", false));
        System.out.println(eval.toMatrixString());
        
        ///////////////////////////////////////
        ThresholdCurve tc = new ThresholdCurve();
        int classIndex = 0;
        Instances result = tc.getCurve(eval.predictions(), classIndex);

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
          public void classifyJ48(double v ) throws Exception   {
       ConverterUtils.DataSource source = new ConverterUtils.DataSource("C:/Users/A.Bagheri/Desktop/Training Dataset.arff");
        Instances dataset = source.getDataSet();
        
        dataset.setClassIndex(dataset.numAttributes() - 1);
        
        System.out.println("SMOTE and Training in 10 Fold Cross validation J48 in Process:");
        // System.out.println(dataset);
        SMOTE smote = new SMOTE();
        smote.setPercentage(26);
        J48 j48 = new J48();
        j48.setConfidenceFactor((float) v);
		  // Print header and instances.

        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(smote);
        fc.setClassifier(j48);

        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(fc, dataset, 10, new Random(1));

        System.out.println(eval.toSummaryString("Results Test:\n", false));
        System.out.println(eval.toMatrixString());
        
        ///////////////////////////////////////
        ThresholdCurve tc = new ThresholdCurve();
        int classIndex = 0;
        Instances result = tc.getCurve(eval.predictions(), classIndex);

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
           public void RandomForrest(int numberoftrees, int numberOfFeatures ) throws Exception   {
            
       ConverterUtils.DataSource source = new ConverterUtils.DataSource("C:/Users/A.Bagheri/Desktop/Training Dataset.arff");
        Instances dataset = source.getDataSet();
        dataset.setClassIndex(dataset.numAttributes() - 1);
        
        System.out.println("SMOTE and Training in 10 Fold Cross validation in Process:");
        // System.out.println(dataset);
        SMOTE smote = new SMOTE();
        smote.setPercentage(26);
        RandomForest RF = new RandomForest();
        RF.setNumTrees(numberoftrees);
        RF.setNumFeatures(numberOfFeatures);
		  // Print header and instances.

        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(smote);
        fc.setClassifier(RF);

        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(fc, dataset, 10, new Random(1));

        System.out.println(eval.toSummaryString("Results Test:\n", false));
        System.out.println(eval.toMatrixString());
        
        ///////////////////////////////////////
        
        ThresholdCurve tc = new ThresholdCurve();
        int classIndex = 0;
        Instances result = tc.getCurve(eval.predictions(), classIndex);

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
