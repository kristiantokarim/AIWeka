/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaai;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.evaluation.output.prediction.PlainText;
import weka.core.Attribute;
import weka.core.Debug.Random;
import weka.core.DenseInstance;
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
    public static Instances data, testSet;
    public static NaiveBayesUpdateable cls;
    public static Classifier cl, model;
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
    public static Classifier readModel(String path) throws Exception {
        return (Classifier) SerializationHelper.read(path);
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
    public static void saveFile(String path) throws Exception{
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(cls);
            oos.flush();
        }
    }
    public static Instances readInstance(Instances ins, Scanner in){
        int numAtt =  ins.numAttributes();
        Instances testSet1 = new Instances(ins);
        Instance newData = new DenseInstance(numAtt);
        String b, ans = "Y";
        double a;
        boolean cont = true;
        testSet1.clear();
        newData.setDataset(testSet1);
        while (ans.equals("Y")){
            System.out.println("\nMasukkan nilai atribut :");
            for (int i=0; i<numAtt;i++){
                String type = (Attribute.typeToString(ins.attribute(i)));
                System.out.print(" >"+(i+1)+". "+ins.attribute(i).name()+": ");
                if ("numeric".equals(type)) {
                    a = in.nextDouble();
                    newData.setValue(i, a);
                }
                else{
                    b = in.next();
                    newData.setValue(i, b);
                }
            }
        testSet1.add(newData);
        System.out.print("\nApakah Anda ingin memasukkan data lain (Y/N)?: ");
        ans = in.next();
        }
        return testSet1;
    }
    public static void reEvaluateModel(Instances test, Classifier modelSet, Instances trainSet) throws Exception{
        Evaluation eval = new Evaluation(trainSet);
        StringBuffer predsBuffer = new StringBuffer();
        PlainText plainText = new PlainText();
        plainText.setHeader(trainSet);
        plainText.setBuffer(predsBuffer);
        
        eval.evaluateModel(modelSet,test, plainText);
        System.out.println("\nRESULT\n");
        System.out.println("=== Predictions on user test set ===\n" +"\n" +"    inst#     actual  predicted error prediction");
        System.out.println(predsBuffer.toString());
        System.out.println(eval.toSummaryString("\n=== Summary ===\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString()); 
    }
    public static Evaluation crossValidation10(Instances ins) throws Exception {
        /*Instances randomData = new Instances(ins);
        randomData.randomize(new Random());
        */
        Evaluation eval = new Evaluation(ins);
        
        //for (int i=0; i<10;i++){
          //  Instances train = randomData.trainCV(10,i);
            //Instances test = randomData.testCV(10,i);
            cls = new NaiveBayesUpdateable();
            cls.buildClassifier(ins);
            //eval.evaluateModel(cls, test);
        //}
        
        eval.crossValidateModel(cls, ins, 10, new Random(1));
        return eval;
    }
    public static void start() throws Exception {
        boolean restart;
        String p;
        do
        {
            restart = false;
            int pil;
            boolean readData = false;
            Scanner in = new Scanner(System.in);
            System.out.println("\t\t\tTugas Kecil AI - WEKA");
            System.out.println("Menu : ");
            System.out.println("1. Read data set");
            System.out.println("2. Read model");
            do {
                System.out.print("Masukan pilihan : ");
                pil = in.nextInt();
                if (pil < 1 || pil > 2) {
                    System.out.println("Input salah !");
                }
            }
            while(pil < 1 || pil > 2);
            System.out.print("Masukan path dari file : ");
            String modelPath = in.next();
            switch (pil) {
                case 1:
                    {
                        
                        readDataFromFile(modelPath);
                        readData = true;
                        break;
                    }
                case 2:
                    {
                        System.out.println("\t\t\tRESULT");
                        System.out.println(readModel(modelPath).toString());
                        break;
                    }
                default:
                    break;
            }
            if (readData) {
                System.out.println("Menu : ");
                System.out.println("1. Filtering with Discretize");
                System.out.println("2. Filtering with NumericToNominal");
                System.out.println("2. Without Filtering");
                do {
                    System.out.print("Masukan pilihan : ");
                    pil = in.nextInt();
                    if (pil < 1 || pil > 3) {
                        System.out.println("Input salah !");
                    }
                }
                while(pil < 1 || pil > 3);
                Instances filtered = new Instances(data);
                switch (pil) {
                    case 1:
                        filtered = discretizeData();
                        filter = ".Discretize-B10-M-1.0-Rfirst-last";
                        break;
                    case 2:
                        filtered = numericToNominalData();
                        filter = ".NumericToNominal-Rfirst-last";
                        break;
                    case 3:
                        filtered = data;
                        filter = "";
                        break;
                    default:
                        break;
                }
                System.out.println("Menu : ");
                System.out.println("1. Create Classification Model");
                System.out.println("2. Evaluate test set based on model");
                do {
                    System.out.print("Masukan pilihan : ");
                    pil = in.nextInt();
                    if (pil < 1 || pil > 2) {
                        System.out.println("Input salah!");
                    }
                }while(pil < 1 || pil > 2);
                if (pil==2){
                    System.out.print("Masukan path model: ");
                    String modPath = in.next();
                    model = readModel(modPath);
                    System.out.println("Masukan test set: ");
                    testSet = readInstance(data, in);
                    reEvaluateModel(testSet, model, filtered);

                }
                else if (pil == 1) {
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
                                + "Relation:\t\tiris-weka.filters.unsupervised.attribute"+filter+"\n"
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
                        Evaluation evCV = crossValidation10(filtered);
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
                        System.out.println(evCV.toSummaryString("\n=== Summary ===\n", false));
                        System.out.println(evCV.toClassDetailsString());
                        System.out.println(evCV.toMatrixString());
                    }
                    System.out.println("Save model ? (Y/N)");
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
                        System.out.print("Masukan path tujuan (isi dengan '-' bila akan disimpan pada folder default):");
                        String pathFile = in.next();
                        String fullName =  (new SimpleDateFormat("dd-MM-yyyy_HHmmss").format(Calendar.getInstance().getTime()))+".model";
                        if (!"-".equals(pathFile)){
                            pathFile += "\\" + fullName;
                        }else
                            pathFile = fullName;
                        saveFile(pathFile);    
                    }
                }
           
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
        }while(restart);
    }
    public static void main(String[] args) throws Exception {
        start();
    }
}