/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka;

import java.awt.BorderLayout;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.ThresholdCurve;
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
public class NaiveBayesCSf {

    public static void main(String args[]) throws Exception {
        
        

       // for(int numberoftrees = 100;numberoftrees <= 400 ;numberoftrees=+100 ){
         //   for(int numberOfFeatures = 5;numberOfFeatures <= 20 ;numberOfFeatures=+5 ){
            
        NaiveBayesCSf RFGSW = new NaiveBayesCSf();
        RFGSW.classifyme();
        
        /*
        for(int numberoftrees = 100;numberoftrees <= 400 ;numberoftrees=+100 ){
          for(int numberOfFeatures = 5;numberOfFeatures <= 20 ;numberOfFeatures =+ 5 ){
              RFGSW.classifyme(numberoftrees, numberOfFeatures);
        }
          } */
    
    
        
        
    }
        
        public void classifyme() throws Exception   {
            
       ConverterUtils.DataSource source = new ConverterUtils.DataSource("C:/Users/A.Bagheri/Desktop/BIProject/GSW9.arff");
        Instances dataset = source.getDataSet();
        dataset.setClassIndex(dataset.numAttributes() - 1);
        
        System.out.println("SMOTE and Training in 10 Fold Cross validation in Process:");
        // System.out.println(dataset);
        SMOTE smote = new SMOTE();
        smote.setPercentage(26);
        NaiveBayes RF = new NaiveBayes();
        
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

    
    

            
        
                
            
    
                

/*

 FilteredClassifier hb = new  FilteredClassifier();
       
 SMOTE s = new SMOTE();
 Classifier bjn = new weka.classifiers.trees.J48();
 Filter fn  = new weka.filters.supervised.attribute.Discretize();
 // protected FilteredClassifier getFilteredClassifier() {
 FilteredClassifier	result;
 Filter		filter;
 Classifier		cls;
  
 result = new FilteredClassifier();
  
 // set filter
 filter = filter.
 result.setFilter(filter);
 FilteredClassifier	result;
 Filter		filter;
 Classifier		cls;
  
 result = new FilteredClassifier();
  
 // set filter
 filter = getFilter();
 result.setFilter(filter);
 }
 */


