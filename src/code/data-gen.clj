(ns skewing.data-gen
  (:require [skewing.core :as core]
            [skewing.trees :as trees]
            [clojure.string :as string]))
(use 'print.foo)
(use 'clojure.java.io)
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
(defn generate-boolean-data
  [bool-fun k]
  "generates data points with bool-fun conjed

   bool-fun is a function that acts on a seq of booleans, returning boolean
   k is the number of booleans"
  (let [truth (possible-truth k)]
    (map #(conj % (apply bool-fun %)) truth)))



(defn generate-random-boolean-function
  "implements a size-n boolean function as a lookup table"
  [n]
  (let [truth (possible-truth n)
        mp (into {} (for [t truth]
                      [t (core/rand-bool)]))]
    (fn [& x] (get mp (take n x)))))

(defn writeln
  [writer s]
  (.write writer (str s \newline)))

(defn generate-data-set
  "n: number of features, k: number of instances"
  [n k]
  (let [name (str n "_" (quot k 2) "_" (System/currentTimeMillis))]
    (with-open [train (writer (str "/home/gabe/Projects/Research2015/data/generated/" name "_train.arff"))
                test (writer (str "/home/gabe/Projects/Research2015/data/generated/" name "_test.arff"))]
      (let [fun (generate-random-boolean-function n)
            train-data (core/sample-with-replacement (generate-boolean-data fun n) (quot k 2))
            test-data (core/sample-with-replacement (generate-boolean-data fun n) (quot k 2))]
        (do (writeln train (str "@relation data-set-" name "_train"))
            (writeln test (str "@relation data-set-" name "_test"))
            (writeln train "@attribute 'class' { true, false}")
            (writeln test "@attribute 'class' { true, false}")
            (doseq [i (range n)]
              (writeln train (str "@attribute 'f" i "' { true, false}"))
              (writeln test (str "@attribute 'f" i "' { true, false}")))
            (writeln train "@data")
            (writeln test "@data")
            (doseq [d train-data]
              (writeln train (string/join "," d)))
            (doseq [d test-data]
              (writeln test (string/join "," d))))))))

"(doseq [i (range 9 19 2)]
  (doseq [j (range 10)]
    (generate-data-set i 500)))"

(defn test-all-have-250
  [direc]
  (let [fs (file-seq (clojure.java.io/file direc))
        k-files (map #(.getName %) fs)
        training-sets (map #(core/get-instances (core/get-data %)) k-files)]
    (for [trn training-sets]
      [trn (count trn)])))
(defn k-size-frequencies
  [dirct]
  (frequencies (for [fl (file-seq (clojure.java.io/file dirct))]
                 (-> fl .getName (string/split #"_") first))))
(defn accuracy-size-k
  [direc k]
  (let [fs (file-seq (clojure.java.io/file direc))
        k-files (filter (fn [x] (= (str k)
                                   (first (clojure.string/split (.getName x) #"_")))) fs)
        test-names (map #(.getPath %) (filter #(= "test.arff" (last (string/split (.getName %) #"_"))) k-files))
        train-names (map #(.getPath %) (filter #(= "train.arff" (last (string/split (.getName %) #"_"))) k-files))
        tupls (for [tst test-names trn train-names :when (= (rest (reverse (string/split trn #"_")))
                                                            (rest (reverse (string/split tst #"_"))))]
                [(core/get-instances (core/get-data trn))
                 (core/get-instances (core/get-data tst))])]
    (do (doseq [[train test] tupls]
          (.setClassIndex train 0)
          (.setClassIndex test 0))
        (for [[train test] tupls]
          (let [decision-tree (trees/build-decision-tree train)
                random-forest (trees/build-random-forest train 10)
                skewed-decision-tree (trees/build-skew-decision-tree train 10 2)
                skewed-random-forest (trees/build-skew-random-forest train 10 2)]
            {:decision-tree (for [inst test] (list (.stringValue inst (core/get-class inst))
                                                   (str (trees/dt-classify inst decision-tree)))),
             :random-forest (for [inst test] (list (.stringValue inst (core/get-class inst))
                                                   (trees/rf-classify inst random-forest))),
             :decision-tree-skew (for [inst test] (list (.stringValue inst (core/get-class inst))
                                                        (trees/rf-classify inst skewed-decision-tree))),
             :random-forest-skew (for [inst test] (list (.stringValue inst (core/get-class inst))
                                                        (trees/rf-classify inst skewed-random-forest)))})))))

(defn summary-statistics
  [result]
  (let [dt-accs (for [run result]
                  (/ (get (frequencies (map #(= (first %) (second %)) (:decision-tree run)))
                          true)
                     (count (:decision-tree run))))
        rf-accs (for [run result]
                  (/ (get (frequencies (map #(= (first %) (core/find-max-key (second %))) (:random-forest run)))
                          true)
                     (count (:random-forest run))))
        dt-skew-accs (for [run result]
                       (/ (get (frequencies (map #(= (first %) (core/find-max-key (second %)))
                                                 (:decision-tree-skew run)))
                               true)
                          (count (:decision-tree-skew run))))
        rf-skew-accs (for [run result]
                       (/ (get (frequencies (map #(= (first %) (core/find-max-key (second %)))
                                                 (:random-forest-skew run)))
                               true)
                          (count (:random-forest-skew run))))]
    {:dt {:accs dt-accs
          :mean-accuracy (float (core/mean dt-accs))
          :variance (float (core/variance dt-accs))}
     :rf {:accs rf-accs
          :mean-accuracy (float (core/mean rf-accs))
          :variance (float (core/variance rf-accs)),}
     :dt-skew {:accs dt-skew-accs
               :mean-accuracy (float (core/mean dt-skew-accs))
               :variance (float (core/variance dt-skew-accs))}
     :rf-skew {:accs rf-skew-accs
               :mean-accuracy (float (core/mean rf-skew-accs))
               :variance (float (core/variance rf-skew-accs))}}))
