function x = mash (a, b)
% MASH uses MESHGRID to return 2 matrices composed of input vectors. The 
% positions of the input vectors are swapped and each element of the output 
% matrices are multiplied with each other.
%
  [p, q] = meshgrid (b, a);
  x = p .* q;
end