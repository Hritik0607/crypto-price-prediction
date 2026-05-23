
use actix_web::{ HttpServer, App, web, Responder, HttpResponse };
use serde::Deserialize;
use prediction_api::get_prediction_from_elasticsearch;

async fn health() -> impl Responder {
    println!("Health endpoint was hit!");
    HttpResponse::Ok().body("Hey there!")
}

#[derive(Deserialize)]
struct RequestParams {
    pair: String
}

/// This is the brain of this REST API
/// Steps:
/// 1. Reads for which crypto pair the client wants predictions
/// 2. Fetches the latest prediction for that pair from Elasticsearch
/// 3. Returns the prediction as JSON
async fn predict(args: web::Query<RequestParams>) -> Result<HttpResponse, actix_web::Error> {
    println!("Requested price prediction for {}", args.pair);
    
    // pair for which we need to fetch the latest prediction
    let pair = &args.pair;

    // TODO:
    // cargo add elasticsearch
    let prediction_output = get_prediction_from_elasticsearch(&pair)?;
    
    // TODO: Format the response as JSON
    // Check actix-web documentation to cast the PredictionOutput as a JSON and return it.
    Ok(HttpResponse::Ok().body("Hey there!"))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Hello, world!");
    
    HttpServer::new(|| {
        App::new()
            .route("/health", web::get().to(health))
            .route("/predict", web::get().to(predict))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
