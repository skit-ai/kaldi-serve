## ASPIRE Chain Model example

This is a basic example using the Aspire recipe, that will show you how to serve a [Kaldi](https://github.com/kaldi-asr/kaldi/) ASR model via [kaldi-serve](https://github.com/Vernacular-ai/kaldi-serve). If you have any query regarding the technical details of the recipe, please check [here](https://github.com/kaldi-asr/kaldi/tree/master/egs/aspire).

### Setup

The setup script will download the chain model (if needed) and format the files as per our requirements:

```bash
./utils/setup_aspire_chain_model.sh --kaldi-root [KALDI ROOT]
```

### Serving the model

You can also run the following script directly as it will call the setup script anyhow to validate the model directory structure:

```bash
./run_server.sh --kaldi-root [KALDI ROOT]
```
