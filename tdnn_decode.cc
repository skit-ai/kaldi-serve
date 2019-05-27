#include <Python.h>
#include <string>

#include "tdnn_wrapper.h"

#define CAPSULE_NAME "TDNN_DECODER_MODEL"

static void capsule_destructor(PyObject* capsule) {
  kaldi::Model *model = (kaldi::Model*)PyCapsule_GetPointer(capsule, CAPSULE_NAME);
  delete model;
}

static PyObject* load_model(PyObject *self, PyObject *args) {
  kaldi::BaseFloat beam;
  int32 max_active;
  int32 min_active;
  kaldi::BaseFloat lattice_beam;
  kaldi::BaseFloat acoustic_scale;
  int32 frame_subsampling_factor;
  char* word_syms_filename;
  char* model_in_filename;
  char* fst_in_str;
  char* mfcc_config;
  char* ie_conf_filename;

  if (!PyArg_ParseTuple(
      args,
      "fiiffisssss",
      &beam, &max_active, &min_active, &lattice_beam,&acoustic_scale, &frame_subsampling_factor,
      &word_syms_filename, &model_in_filename, &fst_in_str,&mfcc_config, &ie_conf_filename
    )
  ) return NULL;

  kaldi::Model* model = (kaldi::Model*) PyMem_Malloc(sizeof(kaldi::Model*));
  model = new kaldi::Model(beam, max_active, min_active, lattice_beam, acoustic_scale,
    frame_subsampling_factor, word_syms_filename, model_in_filename, fst_in_str,
    mfcc_config, ie_conf_filename
  );

  PyObject* model_py = PyCapsule_New((void*)model, CAPSULE_NAME, capsule_destructor);

  if (!model_py)
    return NULL;

  return model_py;
}

static PyObject* infer(PyObject* self, PyObject* args) {
  PyObject* model_py;
  char* wav_file_path;
  int32 max_alternatives;

  if (!PyArg_ParseTuple(args, "Osi", &model_py, &wav_file_path, &max_alternatives)) return NULL;

  kaldi::Model* model = (kaldi::Model*)PyCapsule_GetPointer(model_py, CAPSULE_NAME);
  std::vector<kaldi::Model::result_tuple> results = model->CInfer(wav_file_path, max_alternatives);

  PyObject *py_results = PyList_New(0);

  for (std::vector<kaldi::Model::result_tuple>::const_iterator i = results.begin(); i != results.end(); ++i) {
    PyObject *res = Py_BuildValue(
      "sd", (char*)(std::get<0>(*i).c_str()), std::get<1>(*i)
    );
    PyList_Append(py_results, res);
  }

  return Py_BuildValue("O", py_results);
}

static PyObject* infer_object(PyObject* self, PyObject* args) {
  PyObject* model_py;
  char* wav_file_path;
  int32 max_alternatives;

  if (!PyArg_ParseTuple(args, "Osi", &model_py, &wav_file_path, &max_alternatives)) return NULL;

  kaldi::Model* model = (kaldi::Model*)PyCapsule_GetPointer(model_py, CAPSULE_NAME);
  return Py_BuildValue("i", 1);
}


// Our Module's Function Definition struct
// We require this `NULL` to signal the end of our method definition
static PyMethodDef moduleMethods[] = {
  {"load_model", load_model, METH_VARARGS, "Loads TDNN Model"},
  {"infer", infer, METH_VARARGS, "Converts audio to text"},
  {"infer_object", infer_object, METH_VARARGS, "Starts online decoding"},
  {NULL, NULL, 0, NULL}
};

// Our Module Definition struct
static struct PyModuleDef tdnnDecode = {
  PyModuleDef_HEAD_INIT,
  "tdnn_decode",
  "Kaldi bindings for online TDNN decode",
  -1,
  moduleMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_tdnn_decode(void) {
  return PyModule_Create(&tdnnDecode);
}
