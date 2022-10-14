// See https://aka.ms/new-console-template for more information

using Microsoft.ML;
using Microsoft.ML.AutoML;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json.Nodes;

Console.WriteLine(args[0]);
Console.WriteLine(args[1]);

var mlflowTrackingUri = Environment.GetEnvironmentVariable("MLFLOW_TRACKING_URI") + "/";
var token = Environment.GetEnvironmentVariable("AZUREML_RUN_TOKEN");
var baseUri = new Uri(mlflowTrackingUri!.Replace("azureml:", "https:"));
var listExperimentsUri = new Uri(baseUri, "api/2.0/mlflow/experiments/list?view_type=ALL");
var runId = Environment.GetEnvironmentVariable("MLFLOW_RUN_ID");
var timeStamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

Console.WriteLine(listExperimentsUri);
var httpClient = new HttpClient();
httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {token}");
var response = await httpClient.GetAsync(listExperimentsUri);
Console.WriteLine(response.StatusCode.ToString());
Console.WriteLine(await response.Content.ReadAsStringAsync());

// print al env
var envs = Environment.GetEnvironmentVariables();
foreach(var kv in envs.Keys)
{
    if(kv != null && envs.Contains(kv))
        Console.WriteLine($"{kv?.ToString()}:{envs[kv!]}");
}
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
            .SetTrainingTimeInSeconds(3);

var result = experiment.Run();
context.Model.Save(result.Model, data.Schema, model);
Console.WriteLine($"best metric: {result.Metric}");
var logMetricUri = new Uri(baseUri, $"api/2.0/mlflow/runs/log-metric");
var jsonObject = new JsonObject();
jsonObject["run_id"] = runId;
jsonObject["key"] = "AUC";
jsonObject["value"] = result.Metric;
jsonObject["timestamp"] = timeStamp;
jsonObject["step"] = 0;
var json = jsonObject.ToJsonString();
var httpContent = JsonContent.Create(jsonObject, mediaType: MediaTypeHeaderValue.Parse("application/json"));
var res = await httpClient.PostAsync(logMetricUri, httpContent);
Console.WriteLine(json);
Console.WriteLine(logMetricUri.ToString());
Console.WriteLine(res.StatusCode.ToString());
Console.WriteLine(await res.Content.ReadAsStringAsync());
