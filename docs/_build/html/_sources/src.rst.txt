src package
===========

Submodules
----------

src.activation module
---------------------

.. automodule:: src.activation
   :members:
   :undoc-members:
   :show-inheritance:

src.convolution module
----------------------
We tried to vectorize our convolutions to the maximum, prioritizing the performance.

It implies creating special views of our array, by using the `numpy.lib.stride_tricks`
functions. `sliding_window_view` is the easiest to understand, while maybe not the
fastest compared to `as_strided` (but maybe less risky too).

The calculations are done using `np.einsum`, which is relatively easy to understand
and use. The key relies in understanding the shapes of your inputs/outputs.

Shape
-----
Reminder for 1D:
   - input : ndarray (batch, length, chan_in)
   - d_out : ndarray (batch, length, chan_in) == input.shape
   - X_view : ndarray (batch, out_length, chan_in, self.k_size)
   - delta : ndarray (batch, out_length, chan_out)
   - _gradient["weight"] : ndarray (k_size, chan_in, chan_out)
   - _parameters["weight"] : ndarray (k_size, chan_in, chan_out)

Notes
-----
Notation used for `np.einsum`:
   - b : batch_size
   - w : width (2D) / length (1D)
   - h : height (2D)
   - o : out_width (2D) / out_length (1D)
   - p : out_height (2D)
   - c : chan_in
   - d : chan_out
   - k : k_size (ij for 2D)

Examples
--------
Quick demonstration of `sliding_window_view` in 1D:
.. code-block:: python
   >>> batch, length, chan_in, k_size = 1, 8, 1, 3
   >>> input = np.random.randn(batch, length, chan_in)
   >>> input
   array([[[-0.41982262],
         [ 1.10111123],
         [-0.41115195],
         [ 1.18733225],
         [-1.93463567],
         [-0.22472025],
         [-0.30581971],
         [ 0.40578667]]])

   >>> window = np.lib.stride_tricks.sliding_window_view(input, (1, k_size, chan_in))
   >>> window
   array([[[[[[-0.41982262],
            [ 1.10111123],
            [-0.41115195]]]],
         [[[[ 1.10111123],
            [-0.41115195],
            [ 1.18733225]]]],
      ...

How to deal with stride != 1?
.. code-block:: python
   >>> stride = 3
   >>> window = np.lib.stride_tricks.sliding_window_view(input, (1, k_size, chan_in))[::1, ::stride, ::1]
   >>> window
   array([[[[[[-0.41982262],
            [ 1.10111123],
            [-0.41115195]]]],
         [[[[ 1.18733225],
            [-1.93463567],
            [-0.22472025]]]]]])

Then it is just a matter of reshape, to drop unnecessaries dimensions, e.g. :
.. code-block:: python
   >>> window = window.reshape(batch, out_length, chan_in, k_size)
   >>> window
   array([[[[-0.41982262,  1.10111123, -0.41115195]],
         [[ 1.18733225, -1.93463567, -0.22472025]]]])

And voilà!

.. automodule:: src.convolution
   :members:
   :undoc-members:
   :show-inheritance:

src.encapsulation module
------------------------

.. automodule:: src.encapsulation
   :members:
   :undoc-members:
   :show-inheritance:

src.linear module
-----------------

.. automodule:: src.linear
   :members:
   :undoc-members:
   :show-inheritance:

src.loss module
---------------

.. automodule:: src.loss
   :members:
   :undoc-members:
   :show-inheritance:

src.module module
-----------------

.. automodule:: src.module
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: src
   :members:
   :undoc-members:
   :show-inheritance:
