/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaai;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.core.Instance;
import weka.core.SerializationHelper;
/**
 *
 * @author Kris
 */
public class WekaAI {

    /**
     * @throws java.lang.Exception
     */
    public static Instances data;
    public static NaiveBayesUpdateable cls;
    public static Classifier cl;
    public static String filter;

    /**
     *
     * @param path
     * @throws Exception
     */
    public static void readDataFromFile(String path) throws Exception {
        DataSource source;
        source = new DataSource(path);
        data = source.getDataSet();
        if (data.classIndex() == -1){
            data.setClassIndex(data.numAttributes() - 1);
        }
    }
    public static Instances discretizeData(/*String[] options*/) throws Exception {
        Discretize discretize = new Discretize();
        //discretize.setOptions(options);
        discretize.setInputFormat(data);
        return Filter.useFilter(data, discretize);
    }
    public static Instances numericToNominalData(/*String[] options*/) throws Exception {
        NumericToNominal numericToNominal = new NumericToNominal();
        //numericToNominal.setOptions(options);
        numericToNominal.setInputFormat(data);
        return Filter.useFilter(data, numericToNominal);
    }
    public static String readModel(String path) throws Exception {
        return SerializationHelper.read(path).toString();
    }
    
    /**
     *
     * @param inInstances
     * @return 
     * @throws java.lang.Exception
     */
    
    public static Evaluation fullTrainingNaiveBayes(Instances inInstances) throws Exception {
        cls = new NaiveBayesUpdateable();
        cls.buildClassifier(inInstances);
        Instance current;
        for (int i = 0 ; i < inInstances.numInstances(); i++) {
            current = inInstances.get(i);
            cls.updateClassifier(current);
        }
        Evaluation eval = new Evaluation(inInstances);
        eval.evaluateModel(cls, inInstances);
        return eval;
    }
    public static void start() throws Exception {
        boolean restart;
        do
        {
            restart = false;
            int pil;
            boolean readData = false;
            Scanner in = new Scanner(System.in);
            System.out.println("\t\t\tTugas Kecil AI - WEKA");
            System.out.println("Menu : ");
            System.out.println("1. Read data set");
            System.out.println("2. Read data from user");
            System.out.println("3. Read model");
            do {
                System.out.print("Masukan pilihan : ");
                pil = in.nextInt();
                if (pil < 1 || pil > 3) {
                    System.out.println("Input salah !");
                }
            }
            while(pil < 1 || pil > 3);
            switch (pil) {
                case 1:
                    {
                        System.out.print("Masukan path dari file : ");
                        String modelPath = in.next();
                        readDataFromFile(modelPath);
                        readData = true;
                        break;
                    }
                case 2:
                    //CREATE INSTANCES USER
                    readData = true;
                    break;
                case 3:
                    {
                        System.out.print("Masukan path dari file : ");
                        String modelPath = in.next();
                        System.out.println("\t\t\tRESULT");
                        System.out.println(readModel(modelPath));
                        break;
                    }
                default:
                    break;
            }
            if (readData) {
                System.out.println("Menu : ");
                System.out.println("1. Filtering with discretize");
                System.out.println("2. Filtering with NumericToNominal");
                do {
                    System.out.print("Masukan pilihan : ");
                    pil = in.nextInt();
                    if (pil < 1 || pil > 2) {
                        System.out.println("Input salah !");
                    }
                }
                while(pil < 1 || pil > 2);
                Instances filtered = new Instances(data);
                if (pil == 1) {
                    filtered = discretizeData();
                    filter = "Discretize-B10-M-1.0-Rfirst-last";
                }
                else if (pil == 2){
                    filtered = numericToNominalData();
                    filter = "NumericToNominal-Rfirst-last";
                }
                System.out.println("Menu : ");
                System.out.println("1. Classify with Full-Training");
                System.out.println("2. Classify with 10-Folds CrossValidation");
                do {
                    System.out.print("Masukan pilihan : ");
                    pil = in.nextInt();
                    if (pil < 1 || pil > 2) {
                        System.out.println("Input salah !");
                    }
                }
                while(pil < 1 || pil > 2);
                if (pil == 1) {
                    Evaluation ev = fullTrainingNaiveBayes(filtered);
                    System.out.println("=== Run information ===\n\n"
                            + "Scheme:\t\t\tweka.classifiers.bayes.NaiveBayesUpdateable \n"
                            + "Relation:\t\tiris-weka.filters.unsupervised.attribute."+filter+"\n"
                            + "Instances:\t\t"+filtered.numInstances()+"\n"
                            + "Attributes:\t\t"+filtered.numAttributes());
                    for (int i = 0 ; i < filtered.numAttributes() ; i++) {
                        System.out.println("\t\t\t"+filtered.attribute(i).name());
                    }
                    System.out.println("\n=== Classifier model (full training set) ===\n");
                    System.out.println(cls.toString());
                    System.out.println(ev.toSummaryString("\n=== Summary ===\n", false));
                    System.out.println(ev.toClassDetailsString());
                    System.out.println(ev.toMatrixString());
                }
                else if (pil == 2) {
                    //10-FOLDS CROSSVALIDATION
                }
                System.out.println("Save model ? (Y/N)");
                String p;
                do {
                    System.out.print("Masukan pilihan : ");
                    p = in.next();
                    if (!"y".equals(p.toLowerCase()) && !"n".equals(p.toLowerCase())) {
                        System.out.println("Input salah !");
                    }
                }
                while(!"y".equals(p.toLowerCase()) && !"n".equals(p.toLowerCase()));
                if ("y".equals(p.toLowerCase())) {
                    //SAVE
                }
                System.out.println("Restart?(Y/N)");
                do {
                    System.out.print("Masukan pilihan : ");
                    p = in.next();
                    if (!"y".equals(p.toLowerCase()) && !"n".equals(p.toLowerCase())) {
                        System.out.println("Input salah !");
                    }
                }
                while(!"y".equals(p.toLowerCase()) && !"n".equals(p.toLowerCase()));
                if ("y".equals(p.toLowerCase())) {
                    restart = true;
                }
            }
        }while(restart);
    }
    public static void main(String[] args) throws Exception {
        start();
    }
}