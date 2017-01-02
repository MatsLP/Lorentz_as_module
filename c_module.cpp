#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <arrayobject.h>
#include <vector>
#include <boost/numeric/odeint.hpp>


class observer_class{
    int i;
    PyArrayObject * write_array;
    public: void init(PyArrayObject * np_array){
            i=0;
            write_array = np_array;
                }
    void operator() (const std::vector<double> state, double t ){
        for (int k=0; k<state.size(); k++)
            *(double*)PyArray_GETPTR2(write_array, i, k) = state[k];
        i++;
    }
};

class lorentz_class{
    double sigma;
    double R;
    double b;
    public:
        void init(double in_sigma, double in_R, double in_b){
            sigma = in_sigma;
            R = in_R;
            b = in_b;
        }
        void operator() (const std::vector<double> & x, std::vector<double> & dxdt, const double t){
            dxdt[0] = sigma * ( x[1] - x[0] );
            dxdt[1] = R * x[0] - x[1] - x[0] * x[2];
            dxdt[2] = -b * x[2] + x[0] * x[1];
            dxdt[3] = 1.0; // time component
        }
};

static PyObject* solve (PyObject *dummy, PyObject *args){
    PyObject *out_object;
    PyObject *input_dict;
    PyArrayObject * out_array=NULL;

    double sigma;
    double R;
    double b;
    double dt;
    int numsteps;
    std::vector<double> current_state (4);

    if (!PyArg_ParseTuple(args, "OO", &out_object, &input_dict)) return NULL;
    out_array = (PyArrayObject*)PyArray_FROM_OTF(out_object, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY); // increases reference count

    if (NULL != PyDict_GetItemString(input_dict, "sigma"))
        sigma = PyFloat_AsDouble(PyDict_GetItemString(input_dict, "sigma"));
    else goto fail;
    if (NULL != PyDict_GetItemString(input_dict, "R"))
        R = PyFloat_AsDouble(PyDict_GetItemString(input_dict, "R"));
    else goto fail;
    if (NULL != PyDict_GetItemString(input_dict, "b"))
        b = PyFloat_AsDouble(PyDict_GetItemString(input_dict, "b"));
    else goto fail;
    if (NULL != PyDict_GetItemString(input_dict, "dt"))
        dt = PyFloat_AsDouble(PyDict_GetItemString(input_dict, "dt"));
    else goto fail;
    if (NULL != PyDict_GetItemString(input_dict, "numsteps"))
        numsteps = PyLong_AsLong(PyDict_GetItemString(input_dict, "numsteps"));
    else goto fail;

    lorentz_class lorentz;
    lorentz.init(sigma, R, b);

    observer_class observer;
    observer.init(out_array);

    for (int i=0; i<4; i++)
        current_state[i] = *(double*)PyArray_GETPTR2(out_array, 0, i);
    boost::numeric::odeint::integrate_n_steps(boost::numeric::odeint::make_dense_output( 1.0e-12 , 1.0e-12 , boost::numeric::odeint::runge_kutta_dopri5< std::vector<double> >() ),
                                            lorentz, current_state, 0., dt, numsteps, observer);
    Py_DECREF(out_array); // does not matter if i decrease this or PyObject out_object
    Py_INCREF(Py_None);
    return Py_None;

    fail:
        PyErr_SetString(PyExc_KeyError, "parameters not specified correctly");
        Py_XDECREF(out_array);
        return NULL;
}

// this defines docstring, names of methods and type of arguments when calling
static PyMethodDef mymethods[] = {
    { "solve",solve,
      METH_VARARGS,
      "solve ode"},
    {NULL, NULL, 0, NULL} /* Sentinel */
    };


PyMODINIT_FUNC
initlorentz(void) // name must correspond to final file name
{
   (void)Py_InitModule("lorentz", mymethods);
   import_array();
}
