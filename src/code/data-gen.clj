(ns skewing.data-gen)
(defn cartesian-product
  [a b]
  (for [i a j b]
    [i j]))
(defn possible-truth
  "returns a list of lists of the possible values in an n variable boolean exp"
  [n]
  (if (= n 0)
    '()
    (if (= n 1)
      '([true] [false])
      (if (> n 1)
        (map flatten (reduce cartesian-product (take n (repeat #{true false}))))
        '()))))
(defn boolean-sample
  [bool-fun n k]
  "generates n data points from the set of true values in bool-fun

   bool-fun is a function that acts on a seq of booleans, returning boolean
   n is the number sampled with replacement
   k is the number of booleans"
  (let [pool (filter #(apply bool-fun %) (possible-truth k))]
    (for [i (range n)]
      (rand-nth pool))))

(defn xor
  [a b]
  (or (and (not a) b)
      (and a (not b))))
