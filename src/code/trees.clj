(ns skewing.trees
  (:require [skewing.core :as core]))
(use 'print.foo)

(defn make-node
  [feature]
  {:feature feature, :edges [], :data [], :val nil})

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

(defn set-node-val
  [node val]
  (assoc node :val val))
(defn get-values-of-attr
  [x-attr instances]
  (for [inst instances] (. inst value x-attr)))
(defn probability
  [x-attr] x-attr)
(defn p-y-given-x
  [y x instances])
(defn conditional-entropy
  [x-attr instances]
  (let [num-inst (count instances)
        y-attr (core/get-class-attribute instances)
        x-filters (core/lambdas (if (. x-attr isNumeric)
                                  (core/split-instances x-attr instances)
                                  (core/attribute-vals x-attr))
                                #(. % value x-attr)
                                (if (. x-attr isNumeric) > =)) ;figure out meta data
        y-filters (core/lambdas (if (. y-attr isNumeric)
                                  (core/split-instances y-attr instances)
                                  (core/attribute-vals y-attr))
                                #(. % value y-attr)
                                (if (. y-attr isNumeric) > =))]
    (core/sum
     (for [x x-filters]
       (let [fil-x (filter x instances)
             fil-y (filter x instances)
             pr-x (probability x-attr)
             pr-y (probability y-attr)]
         (* (core/sum
             (for [y y-filters]
               (let [pygx (p-y-given-x y x instances)]
                 (* pygx
                    (core/log2 pygx)))))
            pr-x
            -1))))))
()

;TODO
(defn best-split
  [data possible-features]
  )

;TODO
(defn stop-making-subtree?
  [data]
  (empty? data))

;TODO
(defn majority-vote
  [data]
  )
(defn dt-make-subtree
  [node data possible-features]
  "data: training set, possible-features: set of features that can be split over"
  (if (stop-making-subtree? data)
    (add-data (set-node-val node (majority-vote data))
              data)
    (let [best (best-split data possible-features)]
      (reduce add-edge
              (add-feature node best)
              (for [val (possible-values data best)]
                (make-edge val
                           (dt-make-subtree (make-node nil)
                                            (filter-feature data best val))))))))

(defn build-decision-tree
  [data]
  (dt-make-subtree (add-data (make-node nil) data) data ))
