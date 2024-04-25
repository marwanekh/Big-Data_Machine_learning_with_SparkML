package spark.ML;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.classification.Classifier;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.ChiSqSelector;
import org.apache.spark.ml.feature.ChiSqSelectorModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.HashMap;
import java.util.Map;

import static org.apache.spark.sql.functions.*;

public class Main {
    public static void main(String[] args) {
        //verfier si le nombre d'argument en entre est 2
        if(args.length!=2){
            System.err.println("Usage: Main <TrainFile> <OutputDirectory");
        }

        //Extraire les paramatres d'entres
        String trainFile=args[0];
        String outputDirectory=args[1];

        //creer une session de spark
        SparkSession sparkSession=SparkSession.builder()
                .appName("CardioViscularPrediction")
                .master("local[*]") //dans le cas local
                .getOrCreate();

        sparkSession.sparkContext().setLogLevel("WARN");

        // Définir le schéma des données
        StructType schema = DataTypes.createStructType(new StructField[]{
                DataTypes.createStructField("id", DataTypes.IntegerType, true),
                DataTypes.createStructField("age", DataTypes.IntegerType, true),
                DataTypes.createStructField("education", DataTypes.DoubleType, true),
                DataTypes.createStructField("sex", DataTypes.StringType, true),
                DataTypes.createStructField("is_smoking", DataTypes.StringType, true),
                DataTypes.createStructField("cigsPerDay", DataTypes.DoubleType, true),
                DataTypes.createStructField("BPMeds", DataTypes.DoubleType, true),
                DataTypes.createStructField("prevalentStroke", DataTypes.IntegerType, true),
                DataTypes.createStructField("prevalentHyp", DataTypes.IntegerType, true),
                DataTypes.createStructField("diabetes", DataTypes.IntegerType, true),
                DataTypes.createStructField("totChol", DataTypes.DoubleType, true),
                DataTypes.createStructField("sysBP", DataTypes.DoubleType, true),
                DataTypes.createStructField("diaBP", DataTypes.DoubleType, true),
                DataTypes.createStructField("BMI", DataTypes.DoubleType, true),
                DataTypes.createStructField("heartRate", DataTypes.DoubleType, true),
                DataTypes.createStructField("glucose", DataTypes.DoubleType, true),
                DataTypes.createStructField("TenYearCHD", DataTypes.IntegerType, true)
        });



        //charger les donnes d'entrainement
        Dataset<Row> data=sparkSession.read().option("header",true).schema(schema).csv(trainFile);
        System.out.println("affichage de notre données");
        data.show();


        //1-Data Inspection et Cleaning

        // Supprimer la colonne id
        data = data.drop("id");

        // Calculate the number of null values in each column
        for (String columnName : data.columns()) {
            Dataset<Row> nullCounts = data.agg(
                    sum(when(col(columnName).isNull(), 1).otherwise(0)).alias(columnName + "_null_count")
            );

            // Display the result
            nullCounts.show();
        }


        // Remplacer les valeurs vides dans la colonne glucose par le mode
        Double modeGlucose = data.select("glucose")
                .na().drop().groupBy("glucose").count()
                .orderBy(col("count").desc()).first().getDouble(0);
        data = data.na().fill(modeGlucose, new String[]{"glucose"});

        System.out.println("shape avant supression");
        System.out.println("lignes "+data.count()+" columes "+data.columns().length);

        // Supprimer les lignes avec des valeurs manquantes ou des valeurs aberrantes
        data = data.na().drop()
                .filter(col("totChol").leq(600.0))
                .filter(col("sysBP").leq(295.0));
        System.out.println("shape apres supression");
        System.out.println("lignes "+data.count()+" columes "+data.columns().length);

        // Remplacer les valeurs dans les colonnes sex et is_smoking
        data = data.withColumn("sex", when(col("sex").equalTo("F"), lit(0)).otherwise(lit(1)))
                .withColumn("is_smoking", when(col("is_smoking").equalTo("NO"), lit(0)).otherwise(lit(1)));

        // Enregistrer le DataFrame après prétraitement
        data.write().format("csv").option("header", true).save(outputDirectory+"/cleaned_data");
        //Fin preprocessing



        //2- Exploratory Data Analysis
        // Describe the DataFrame
        Dataset<Row> description = data.describe();

        // Show the summary statistics
        description.show();

        // Calculate the correlation matrix
        // Get the array of column names
        String[] columns = data.columns();

        // Calculate the correlation between all pairs of numeric columns
        for (int i = 0; i < columns.length; i++) {
            for (int j = i + 1; j < columns.length; j++) {
                Double correlation = data.stat().corr(columns[i], columns[j]);
                System.out.println("Correlation between " + columns[i] + " and " + columns[j] + ":");
                System.out.println(correlation);
            }
        }

        //Resampling imbalanced dataset by oversampling positive cases
        // Filtrer les lignes où la colonne 'TenYearCHD' est égale à 1 ou 0
        Dataset<Row> target1 = data.filter(data.col("TenYearCHD").equalTo(1));
        Dataset<Row> target0 = data.filter(data.col("TenYearCHD").equalTo(0));



        // Calculer le nombre d'échantillons à extraire
        long sampleSize = target0.count();


        // Effectuer un échantillonnage sur target1 pour équilibrer les classes
        Dataset<Row> target1Resampled = target1.sample(true, (double) sampleSize / target1.count(), 40L);

        // Afficher le nombre d'échantillons dans chaque classe
        System.out.println("Nombre d'échantillons dans target1 : " + target1Resampled.count());
        System.out.println("Nombre d'échantillons dans target0 : " + target0.count());



        //3-Selection des colonnes interressantes
        // Définir les fonctionnalités et la variable cible
        String[] inputCols= new String[]{"age", "education", "sex", "is_smoking", "cigsPerDay", "BPMeds",
                "prevalentStroke", "prevalentHyp", "diabetes", "totChol", "sysBP", "diaBP", "BMI",
                "heartRate", "glucose"};

        String outputCol = "features";

        // Assembler les fonctionnalités dans un vecteur
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(inputCols)
                .setOutputCol(outputCol);
        Dataset<Row> assembledData = assembler.transform(data);

        // Définir le sélecteur ChiSqSelector
        ChiSqSelector selector = new ChiSqSelector()
                .setNumTopFeatures(10)
                .setFeaturesCol(outputCol)
                .setLabelCol("TenYearCHD"); // Nom de la colonne de la variable cible

        // Appliquer le sélecteur aux données
        ChiSqSelectorModel selectorModel = selector.fit(assembledData);
        Dataset<Row> selectedData = selectorModel.transform(assembledData);

        // Afficher les scores des fonctionnalités
        System.out.println("Scores des fonctionnalités :");
        int[] pValues = selectorModel.selectedFeatures(); // Récupérer les valeurs p
        String[] inputColsAssembeled = assembledData.columns(); // Récupérer les noms des colonnes originales
        for (int i = 0; i < pValues.length; i++) {
            System.out.println("Feature " + inputColsAssembeled[i] + " : " + pValues[i]);
        }

// Récupérer les noms des 10 meilleures fonctionnalités
        int[] selectedFeaturesIndices = selectorModel.selectedFeatures();
        System.out.println("Les 10 meilleures fonctionnalités :");
        for (int index : selectedFeaturesIndices) {
            System.out.println(inputColsAssembeled[index]);
        }

        Dataset<Row> dataImp = data.select("age",
                "prevalentHyp",
                "sysBP",
                "diaBP",
                "glucose",
                "diabetes",
                "BPMeds",
                "sex",
                "education",
                "prevalentStroke",
                "TenYearCHD");
        // Diviser les données en ensembles de formation et de test
        Dataset<Row>[] splits = dataImp.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> trainData = splits[0];
        Dataset<Row> testData = splits[1];

        // Convertir les colonnes en vecteur de features
        VectorAssembler assembler1 = new VectorAssembler()
                .setInputCols(new String[]{"age",
                        "prevalentHyp",
                        "sysBP",
                        "diaBP",
                        "glucose",
                        "diabetes",
                        "BPMeds",
                        "sex",
                        "education",
                        "prevalentStroke"})
                .setOutputCol("features");

        // Préparer les modèles de classification
        Map<String, Classifier> classifiers = new HashMap<>();
        classifiers.put("Logistic Regression", new LogisticRegression());
        classifiers.put("Random Forest", new RandomForestClassifier());
        classifiers.put("Decision Tree", new DecisionTreeClassifier());

        // Entraîner et évaluer les modèles

        // Définir une map pour stocker les scores de chaque modèle
        Map<String, Double> modelScores = new HashMap<>();

        for (Map.Entry<String, Classifier> entry : classifiers.entrySet()) {
            String modelName = entry.getKey();
            Classifier classifier = entry.getValue();
            Dataset<Row> assembledTrainData = assembler1.transform(trainData)
                    .withColumnRenamed("TenYearCHD", "label")
                    .select("features", "label");
            // Entraîner le modèle
            Model model = classifier.fit(assembledTrainData);

            // Convertir les colonnes en vecteur de features
            Dataset<Row> assembledTestData = assembler1.transform(testData);

            // Faire des prédictions sur les données de test
            Dataset<Row> predictions = model.transform(assembledTestData);

            // Évaluer les prédictions
            MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                    .setLabelCol("TenYearCHD")
                    .setPredictionCol("prediction")
                    .setMetricName("accuracy");

            double accuracy = evaluator.evaluate(predictions);
            System.out.println("Accuracy of " + modelName + ": " + accuracy);

            // Stocker le score de chaque modèle dans la map
            modelScores.put(modelName, accuracy);

            // Enregistrer les prédictions dans l'emplacement spécifié
            Dataset<Row> predictionsToSave = predictions.select("age",
                    "prevalentHyp",
                    "sysBP",
                    "diaBP",
                    "glucose",
                    "diabetes",
                    "BPMeds",
                    "sex",
                    "education",
                    "prevalentStroke", "prediction");
            predictionsToSave.write().format("csv").option("header", true).save(outputDirectory + "/" + modelName);

        }

        // finir la session spark
        sparkSession.stop();


    }
}