import boto3
import json

runtime_client = boto3.client('sagemaker-runtime')

ENDPOINT_NAME = "YOUR-XGBOOST-ENDPOINT-NAME" 

def lambda_handler(event, context):
    try:
        if 'body' in event:
            body = json.loads(event['body'])
            features = body.get('features')
        else:
            features = event.get('features')
            
        if not features:
            return {
                "statusCode": 400, 
                "body": json.dumps({"error": "No features provided in the event"})
            }

        csv_payload = ",".join(str(x) for x in features)

        response = runtime_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='text/csv',
            Body=csv_payload
        )
        
        result = response['Body'].read().decode('utf-8')
        prediction_probability = float(result)
        predicted_class = 1 if prediction_probability > 0.5 else 0

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'probability': prediction_probability,
                'predicted_class': predicted_class
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
