/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaai;
import java.io.FileInputStream;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.DecisionStump;
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
    public static Instances discretizeData(String[] options) throws Exception {
        Discretize discretize = new Discretize();
        discretize.setOptions(options);
        return Filter.useFilter(data, discretize);
    }
    public static Instances numericToNominalData(String[] options) throws Exception {
        NumericToNominal numericToNominal = new NumericToNominal();
        numericToNominal.setOptions(options);
        return Filter.useFilter(data, numericToNominal);
    }
    public static String readModel(String path) throws Exception {
        return SerializationHelper.read(path).toString();
    }
    public static void main(String[] args) throws Exception {
        readDataFromFile("C:\\Users\\Kris\\Documents\\NetBeansProjects\\data\\airline.arff");
        System.out.println(readModel("C:\\Users\\Kris\\Documents\\NetBeansProjects\\data\\ds.model"));
    }
}