#include <Python.h>
#include <numpy/arrayobject.h>
#include "blas.h"

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

#define PARSE_ARRAY(o) (PyArrayObject*) PyArray_FROM_OTF(o, NPY_DOUBLE, \
        NPY_INOUT_ARRAY)

static PyObject *icf_update (PyObject *self, PyObject *args)
{
    PyObject *V_obj, *U_obj, *user_items, *item_users;
    if (!PyArg_ParseTuple(args, "OOOO", &U_obj, &V_obj, &user_items,
                          &item_users))
        return NULL;

    PyArrayObject *V_array = PARSE_ARRAY(V_obj),
                  *U_array = PARSE_ARRAY(U_obj);
    if (U_array == NULL || V_array == NULL) goto fail;

    int nusers = (int)PyArray_DIM(U_array, 0),
        nitems = (int)PyArray_DIM(V_array, 0),
        ntopics = (int)PyArray_DIM(V_array, 1);
    if (ntopics != (int)PyArray_DIM(U_array, 1) ||
        nusers != (int)PyList_Size(user_items)) {
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch");
        goto fail;
    }

    double *x = malloc(ntopics*sizeof(double)),
           *U = PyArray_DATA(U_array),
           *V = PyArray_DATA(V_array),
           *fU = malloc(ntopics*nusers*sizeof(double)),
           *fV = malloc(ntopics*nitems*sizeof(double)),
           *UTU = malloc(ntopics*ntopics*sizeof(double)),
           *VTV = malloc(ntopics*ntopics*sizeof(double));

    char n = 'N', t = 'T';
    double one = 1.0, zero = 0.0, a = 0.01, b = 40.0, l2u = 0.0, l2v = 0.0;

    int uid, i, j, l, el, ione = 1, info, *ipiv = malloc(ntopics*sizeof(int));
    for (uid = 0; uid < nusers; ++uid) {
        for (i = 0; i < ntopics*nitems; ++i) fV[i] = a*V[i];
        for (i = 0; i < ntopics; ++i) x[i] = 0.0;

        PyObject *items = PyList_GetItem(user_items, uid);
        l = (int)PyList_Size(items);
        for (i = 0; i < l; ++i) {
            el = PyInt_AsLong(PyList_GetItem(items, i));
            for (j = 0; j < ntopics; ++j) {
                fV[el*ntopics+j] = b*V[el*ntopics+j];
                x[j] += b*V[el*ntopics+j];
            }
        }

        dgemm_(&n, &t, &ntopics, &ntopics, &nitems, &one, V, &ntopics, fV,
               &ntopics, &zero, VTV, &ntopics);

        for (j = 0; j < ntopics; ++j) VTV[j*ntopics+j] += l2u;

        dgesv_(&ntopics, &ione, VTV, &ntopics, ipiv, x, &ntopics, &info);
        for (i = 0; i < ntopics; ++i) U[uid*ntopics+i] = x[i];
    }

    for (uid = 0; uid < nitems; ++uid) {
        for (i = 0; i < ntopics*nusers; ++i) fU[i] = a*U[i];
        for (i = 0; i < ntopics; ++i) x[i] = 0.0;

        PyObject *items = PyList_GetItem(item_users, uid);
        l = (int)PyList_Size(items);
        for (i = 0; i < l; ++i) {
            el = PyInt_AsLong(PyList_GetItem(items, i));
            for (j = 0; j < ntopics; ++j) {
                fU[el*ntopics+j] = b*U[el*ntopics+j];
                x[j] += b*U[el*ntopics+j];
            }
        }

        dgemm_(&n, &t, &ntopics, &ntopics, &nusers, &one, U, &ntopics, fU,
               &ntopics, &zero, UTU, &ntopics);

        for (j = 0; j < ntopics; ++j) UTU[j*ntopics+j] += l2v;

        dgesv_(&ntopics, &ione, UTU, &ntopics, ipiv, x, &ntopics, &info);
        for (i = 0; i < ntopics; ++i) V[uid*ntopics+i] = x[i];
    }

//    // Pre-compute V^T.V.
//    dgemm_(&n, &t, &ntopics, &ntopics, &nitems, &one, V, &ntopics, V,
//           &ntopics, &zero, VTV, &ntopics);
//
//    int i, j, k, l, uid, ione = 1, *ipiv = malloc(ntopics*sizeof(int)), info;
//    long el;
//    double val, alpha = 1.0, l2u = 0.01, l2v = 0.01;
//    for (uid = 0; uid < nusers; ++uid) {
//        PyObject *items = PyList_GetItem(user_items, uid);
//        l = (int)PyList_Size(items);
//
//        double *m = malloc(ntopics*ntopics*sizeof(double)),
//               *b = malloc(ntopics*sizeof(double));
//        for (i = 0; i < ntopics*ntopics; ++i) m[i] = VTV[i];
//        for (i = 0; i < ntopics; ++i) {
//            b[i] = 0.0;
//            m[i*ntopics+i] += l2u;
//        }
//
//        for (i = 0; i < l; ++i) {
//            el = PyInt_AsLong(PyList_GetItem(items, i));
//            for (j = 0; j < ntopics; ++j) {
//                b[j] += (1+alpha)*V[el*ntopics+j];
//                m[j*ntopics+j] += alpha*V[el*ntopics+j]*V[el*ntopics+j];
//                for (k = j+1; k < ntopics; ++k) {
//                    val = alpha*V[el*ntopics+j]*V[el*ntopics+k];
//                    m[j*ntopics+k] += val;
//                    m[k*ntopics+j] += val;
//                }
//            }
//        }
//
//        dgesv_(&ntopics, &ione, m, &ntopics, ipiv, b, &ntopics, &info);
//        for (i = 0; i < ntopics; ++i) U[uid*ntopics+i] = b[i];
//
//        free(b);
//        free(m);
//    }
//
//    // Pre-compute U^T.U.
//    dgemm_(&n, &t, &ntopics, &ntopics, &nusers, &one, U, &ntopics, U,
//           &ntopics, &zero, UTU, &ntopics);
//
//    int iid;
//    for (iid = 0; iid < nitems; ++iid) {
//        PyObject *users = PyList_GetItem(item_users, iid);
//        l = (int)PyList_Size(users);
//
//        double *m = malloc(ntopics*ntopics*sizeof(double)),
//               *b = malloc(ntopics*sizeof(double));
//        for (i = 0; i < ntopics*ntopics; ++i) m[i] = UTU[i];
//        for (i = 0; i < ntopics; ++i) {
//            b[i] = 0.0;
//            m[i*ntopics+i] += l2v;
//        }
//
//        for (i = 0; i < l; ++i) {
//            el = PyInt_AsLong(PyList_GetItem(users, i));
//            for (j = 0; j < ntopics; ++j) {
//                b[j] += (1+alpha)*U[el*ntopics+j];
//                m[j*ntopics+j] += alpha*U[el*ntopics+j]*U[el*ntopics+j];
//                for (k = j+1; k < ntopics; ++k) {
//                    val = alpha*U[el*ntopics+j]*U[el*ntopics+k];
//                    m[j*ntopics+k] += val;
//                    m[k*ntopics+j] += val;
//                }
//            }
//        }
//
//        dgesv_(&ntopics, &ione, m, &ntopics, ipiv, b, &ntopics, &info);
//        for (i = 0; i < ntopics; ++i) V[iid*ntopics+i] = b[i];
//
//        free(b);
//        free(m);
//    }

    free(fU);
    free(fV);
    free(x);
    free(ipiv);
    free(UTU);
    free(VTV);
    Py_DECREF(V_array);
    Py_DECREF(U_array);

    Py_INCREF(Py_None);
    return Py_None;

fail:

    Py_XDECREF(V_array);
    Py_XDECREF(U_array);

    return NULL;
}

static PyMethodDef icf_methods[] = {
    {"update", (PyCFunction)icf_update, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int icf_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int icf_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_icf",
    NULL,
    sizeof(struct module_state),
    icf_methods,
    NULL,
    icf_traverse,
    icf_clear,
    NULL
};

#define INITERROR return NULL

PyObject *PyInit__icf(void)
#else
#define INITERROR return

void init_icf(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("_icf", icf_methods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("_icf.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    import_array();

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
