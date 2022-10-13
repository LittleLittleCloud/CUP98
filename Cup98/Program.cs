// See https://aka.ms/new-console-template for more information

using Microsoft.ML;
using Microsoft.ML.AutoML;

Console.WriteLine(args[0]);
Console.WriteLine(args[1]);

var mlflowTrackingUri = Environment.GetEnvironmentVariable("MLFLOW_TRACKING_URI");
Console.WriteLine(mlflowTrackingUri);
var baseUri = new Uri(mlflowTrackingUri!.Replace("azureml", "https"));
var listExperimentsUri = new Uri(baseUri, "2.0/mlflow/experiments/list");

var httpClient = new HttpClient();
var response = await httpClient.GetAsync(listExperimentsUri);
Console.WriteLine(await response.Content.ReadAsStringAsync());
var label = "TARGET_B";
var cupData = args[0];
var model = args[1];
var context = new MLContext();
context.Log += (o, e) =>
{
    if (e.RawMessage.Contains("Trial"))
    {
        Console.WriteLine(e.RawMessage);
    }
};
var columnInferenceResult = context.Auto().InferColumns(cupData, labelColumnName: label);
var textLoader = context.Data.CreateTextLoader(columnInferenceResult.TextLoaderOptions);
var data = textLoader.Load(cupData);

var experiment = context.Auto().CreateExperiment();
var pipeline = context.Auto().Featurizer(data, columnInferenceResult.ColumnInformation, "__Features__")
                    .Append(context.Auto().BinaryClassification(label, "__Features__"));

experiment.SetDataset(context.Data.TrainTestSplit(data))
            .SetBinaryClassificationMetric(BinaryClassificationMetric.AreaUnderRocCurve, label)
            .SetPipeline(pipeline)
            .SetTrainingTimeInSeconds(30);

var result = experiment.Run();
context.Model.Save(result.Model, data.Schema, model);
Console.WriteLine($"best metric: {result.Metric}");
