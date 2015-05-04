(ns skewing.trees
  (:require [skewing.core :as core]))
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
        x-filters (core/lambdas (if (. x-attr isNumeric)
                                  (core/split-instances x-attr instances)
                                  (core/attribute-vals x-attr))
                                #(. % stringValue x-attr)
                                (if (. x-attr isNumeric) > =)) ;figure out meta data
        y-filters (core/lambdas (if (. y-attr isNumeric)
                                  (core/split-instances y-attr instances)
                                  (core/attribute-vals y-attr))
                                #(. % stringValue y-attr)
                                (if (. y-attr isNumeric) > =))
        x-splits (into {} (for [x x-filters]
                            [(:split (meta x)) (filter x instances)]))]
    [x-splits
     (core/sum
      (for [x x-filters]
        (let [fil-x (get x-splits (:split (meta x)))
              pr-x (px (count fil-x) num-inst x-attr)]
          (* (core/sum
              (for [y y-filters]
                (let [fil-y (filter y instances)
                      pygx (p-y-given-x (count (filter #(and (x %) (y %)) instances))
                                        (count fil-y)
                                        y-attr)]
                  (core/entropy pygx))))
             pr-x
             -1))))]))

(defn best-split
  [instances possible-features]
  (apply max-key #(second (second %)) (for [attr possible-features]
                                        [attr (conditional-entropy attr instances)])))

(defn stop-making-subtree?
  [instances possible-features]
  (or (empty? possible-features)
      (empty? instances)
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
(defn nonempty-splits
  "returns the values that at least one instance takes in the set of instances"
  [instances best-attr x-splits]
  ())
(defn dt-make-subtree
  [node instance-seq possible-features]
  "data: training set, possible-features: set of features that can be split over"
  (if (stop-making-subtree? instance-seq possible-features)
    (-> node
          (set-node-label (majority-class-vote instance-seq))
          (add-data instance-seq))
    (let [best (best-split instance-seq possible-features)
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
