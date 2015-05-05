(ns skewing.trees
  (:require [skewing.core :as core]
            [taoensso.timbre :as timbre]))
(timbre/refer-timbre)

(use 'print.foo)

(defn make-node
  [feature]
  {:feature feature, :edges [], :data [], :label nil})

(defn add-data
  [node data]
  (assoc node :data data))

(defn add-feature
  [node feature]
  (assoc node :feature feature))

(defn make-edge
  [feature-val next-node]
  {:feature-val feature-val, :next-node next-node})

(defn add-edge
  [node edge]
  (assoc node :edges (conj (:edges node) edge)))

(defn set-node-label
  [node label]
  (assoc node :label label))

(defn get-values-of-attr
  [x-attr instances]
  (for [inst instances] (. inst value x-attr)))

(defn px
  "num-x:number of instances with x, "
  [num-x training-size x-attr]
  (if (. x-attr isNominal)
    (/ (+ num-x 1)
       (+ training-size (count (core/attribute-vals x-attr))))
    (throw (Exception. "x-attr must be nominal"))))

(defn p-y-given-x
  [num-y-and-x num-y y-attr]
  (if (. y-attr isNominal)
    (/ (+ num-y-and-x 1)
       (+ num-y (count (core/attribute-vals y-attr))))
    (throw (Exception. "x-attr must be nominal"))))

(defn conditional-entropy
  [x-attr instances]
  (let [num-inst (count instances)
        y-attr (core/get-class-attribute instances)
        x-filters (p :x-filters (core/lambdas (if (. x-attr isNumeric)
                                   (core/split-instances x-attr instances)
                                   (core/attribute-vals x-attr))
                                 #(. % stringValue x-attr)
                                 (if (. x-attr isNumeric) > =))) ;figure out meta data
        y-filters (p :y-filters (core/lambdas (if (. y-attr isNumeric)
                                   (core/split-instances y-attr instances)
                                   (core/attribute-vals y-attr))
                                 #(. % stringValue y-attr)
                                 (if (. y-attr isNumeric) > =)))
        x-splits (p :x-splits (into {} (for [x x-filters]
                                         [(:split (meta x)) (filter x instances)])))
        y-splits (p :y-splits (into {} (for [y y-filters]
                                         [(:split (meta y)) (filter y instances)])))]
    [x-splits
     (core/sum
      (for [x x-filters]
        (let [fil-x (get x-splits (:split (meta x)))
              pr-x (p :pr-x (px (count fil-x) num-inst x-attr))]
          (* (core/sum
              (for [y y-filters]
                (let [fil-y (get y-splits (:split (meta y)))
                      pygx (p :p-y-given-x (p-y-given-x (count (filter #(x %) fil-y))
                                                        (count fil-y)
                                                        y-attr))]
                  (core/entropy pygx))))
             pr-x
             -1))))]))

(defn best-split
  [instances possible-features]
  (apply max-key #(second (second %)) (for [attr possible-features]
                                        [attr (conditional-entropy attr instances)])))

(defn stop-making-subtree?
  [instances possible-features m]
  (or (empty? possible-features)
      (>= m (count instances))
      (let [y-attr (core/get-class instances)]
        (core/same (map #(core/get-attr-val y-attr %) instances)))))

(defn majority-class-vote
  [instances]
  (let [y-attr (. (nth instances 0) classAttribute)
        attr-seq (core/attribute-vals y-attr)]
    (apply max-key second (for [attr-val attr-seq]
                            [attr-val (/ (count (filter #(= attr-val (. % stringValue y-attr))
                                                      instances))
                                         (count instances))]))))

(defn dt-make-subtree
  "data: training set, possible-features: set of features that can be split over"
  [node instance-seq possible-features]
  (if (stop-making-subtree? instance-seq possible-features 2)
    (-> node
        (set-node-label (majority-class-vote instance-seq))
        (add-data instance-seq))
    (let [best (p :find-split (best-split instance-seq possible-features))
          best-attr (first best)
          conditional-entropy (second (second best))
          x-splits (first (second best))]
      (reduce add-edge
              (add-feature node best-attr)
              (for [[val insts] x-splits :when (> (count insts) 0)]
                (make-edge val
                           (dt-make-subtree (make-node nil)
                                            insts
                                            (disj possible-features best-attr))))))))

(defn build-decision-tree
  [instances]
  (if (> (count instances) 0)
    (let [instances (seq instances)]
      (dt-make-subtree (make-node nil) instances (set (core/attribute-seq instances))))
    (throw (Exception. "instances must have at least at least one element"))))


(defn dt-make-randomized-subtree
  [node instance-seq possible-features]
  (if (stop-making-subtree? instance-seq possible-features 2)
    (-> node
        (set-node-label (majority-class-vote instance-seq))
        (add-data instance-seq))
    (let [best (p :find-split (best-split instance-seq (possible-features)))
          best-attr (first best)
          conditional-entropy (second (second best))
          x-splits (first (second best))]
      (reduce add-edge
              (add-feature node best-attr)
              (for [[val insts] x-splits :when (> (count insts) 0)]
                (make-edge val
                           (dt-make-subtree (make-node nil)
                                            insts
                                            (disj possible-features best-attr))))))))
(defn build-random-forest
  "returns a list of t random decision trees"
  [instances t]
  (for [i (range t)]
    (let [new-instances (core/sample-with-replacement instances (count instances))]
      (dt-make-randomized-subtree (make-node nil)
                                  new-instances
                                  (set (core/attribute-seq new-instances))))))
