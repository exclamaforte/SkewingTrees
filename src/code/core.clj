(ns skewing.core
  (:import [java.io FileReader BufferedReader]
           [weka.core Instances]
           [java.lang.Math]
           weka.core.converters.ArffLoader$ArffReader))
(defn get-data
  "Creates a weka Instances class from a string"
  [st]
  (ArffLoader$ArffReader. (BufferedReader. (FileReader. st))))
(defn get-instances
  "gets the instances from a weka reader"
  [rdr]
  (Instances. (. rdr getData)))
(defn nth-instance
  "gets the nth instance from the Instances class"
  [insts n]
  (. insts get n))
(defn nth-attribute
  "gets the nth attribute from an Instance"
  [inst n]
  (. inst attribute n))
(defn get-structure
  "returns a list attributes that is of a dataset, from a Instances class."
  [insts]
  (enumeration-seq (. (nth (. insts toArray) 0) enumerateAttributes)))
(defn get-attr-val
  "returns the value of attr in inst"
  [attr inst]
  (. inst value attr))
(defn get-class
  "get the Attribute class of an instance"
  [insta]
  (. insta classAttribute))

(defn sample-with-replacement
  "takes a collection and a number and builds a new list of n samples from the first list"
  [col n]
  (if (empty? col)
    (empty col)
    (into (empty col)
          (for [i (range n)]
            (rand-nth (seq col))))))

(defn sum
  [coll]
  (reduce + 0 coll))

(defn log2 [n]
  (/ (java.lang.Math/log n) (java.lang.Math/log 2)))

(defn entropy
  [flt]
  "split is a normalized sequence of splits"
  (* flt
     (log2 flt)))
(defn normalize
  [coll]
  (map #(/ % (sum coll))
       coll))
(defn average
  [coll]
  (/ (sum coll)
     (count coll)))
(defn get-class-attribute
  [inst-or-instances]
  (. inst-or-instances classAttribute))
(defn lambdas
  "creates a collection of functions that test whethes something is equal to the corresponding element in the coll"
  [coll accessor func]
  (into (empty coll) (for [i coll]
                       (with-meta (fn [x] (func (accessor x) i)) {:split i}))))
;write some tests
(defn split-collection
  [coll comperator]
  (let [coll (into [] coll)]
    (pop (reduce (fn [prev next]
                   (let [lst (peek prev)
                         splits (pop prev)
                         last-val (comperator lst)
                         next-val (comperator next)]
                     (if (= last-val next-val)
                       (conj splits next)
                       (conj splits (average [last-val next-val]) next))))
                 [(first coll)]
                 (rest coll)))))
(defn split-instances
  "x-attr is a real attribute, returns a vector of all of the split points of x-attr"
  [x-attr instances]
  (if (. x-attr isNumeric)
    (let [instances (sort-by #(. % value x-attr) instances)]
      (split-collection instances #(. % value x-attr)))
    (throw (Exception. (str "Feature " x-attr " must be Numeric")))))
(defn attribute-vals
  "returns the possible vals of an attribute"
  [attr]
  (enumeration-seq (. attr enumerateValues)))
(defn attribute-seq
  "returns an sequence of the attributes"
  [insts]
  (enumeration-seq (. insts enumerateAttributes)))
(def test-data (get-data "/home/gabe/Projects/Research2015/data/lymph_train.arff"))
(def test-instances (get-instances test-data))
(. test-instances setClassIndex (- (. test-instances numAttributes) 1))
(def wine-data (get-data "/home/gabe/Projects/Research2015/data/wine.arff"))
(def wine (get-instances wine-data))
(. wine setClassIndex 0)
(def numeric (filter #(. % isNumeric) (enumeration-seq (. wine enumerateAttributes))))
(def not-numeric (filter #(not (. % isNumeric)) (enumeration-seq (. wine enumerateAttributes))))
