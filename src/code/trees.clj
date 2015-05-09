(ns skewing.trees
  (:require [skewing.core :as core]
            [skewing.skew :as skew]
            [taoensso.timbre :as timbre]
            [bigml.sampling.simple :as simple]))
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
  [feature-val function next-node]
  {:feature-val feature-val, :function function, :next-node next-node})

(defn add-edge
  [node edge]
  (assoc node :edges (conj (:edges node) edge)))

(defn add-label
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
                            [x (filter x instances)]))
        y-splits (into {} (for [y y-filters]
                            [y (filter y instances)]))]
    [x-splits
     (core/sum
      (for [x x-filters]
        (let [fil-x (get x-splits x)
              pr-x (p :pr-x (px (count fil-x) num-inst x-attr))]
          (* (core/sum
              (for [y y-filters]
                (let [fil-y (get y-splits y)
                      pygx (p-y-given-x (count (filter #(x %) fil-y))
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
        (add-label (majority-class-vote instance-seq))
        (add-data instance-seq))
    (let [best (best-split instance-seq possible-features)
          best-attr (first best)
          conditional-entropy (second (second best))
          x-splits (first (second best))]
      (if (> 0 conditional-entropy)
        (-> node
            (add-label (majority-class-vote instance-seq))
            (add-data instance-seq))
        (reduce add-edge
                (add-feature node best-attr)
                (for [[val insts] x-splits]
                  (if (empty? insts)
                    (make-edge (:split (meta val))
                               val
                               (add-label (make-node nil) (majority-class-vote instance-seq)))
                    (make-edge (:split (meta val))
                               val
                               (dt-make-subtree (make-node nil)
                                                insts
                                                (disj possible-features best-attr))))))))))

(defn build-decision-tree
  [instances]
  (if (> (count instances) 0)
    (let [instance-seq (seq instances)]
      (dt-make-subtree (make-node nil) instance-seq (set (core/attribute-seq instance-seq))))
    (throw (Exception. "hue"))))

(defn dt-make-randomized-subtree
  [node instance-seq possible-features]
  (if (stop-making-subtree? instance-seq possible-features 2)
    (-> node
        (add-label (majority-class-vote instance-seq))
        (add-data instance-seq))
    (let [best (best-split instance-seq (take (+ (quot (count possible-features) 2) 1)
                                              (simple/sample possible-features)))
          best-attr (first best)
          conditional-entropy (second (second best))
          x-splits (first (second best))]
      (if (> 0 conditional-entropy)
        (-> node
            (add-label (majority-class-vote instance-seq))
            (add-data instance-seq))
        (reduce add-edge
                (add-feature node best-attr)
                (for [[val insts] x-splits]
                  (if (empty? insts)
                    (make-edge (:split (meta val))
                               val
                               (add-label (make-node nil) (majority-class-vote instance-seq)))
                    (make-edge (:split (meta val))
                               val
                               (dt-make-randomized-subtree (make-node nil)
                                                           insts
                                                           (disj possible-features best-attr))))))))))
(defn build-random-forest
  "returns a list of t random decision trees"
  [instances t]
  (for [i (range t)]
    (let [new-instances (core/sample-with-replacement instances (count instances))]
      (dt-make-randomized-subtree (make-node nil)
                                  new-instances
                                  (set (core/attribute-seq new-instances))))))
(defn dt-classify
  "returns the class of instance in tree"
  [instance tree]
  (if (nil? tree)
    (do (print tree)
        (throw (Exception. "tree is nil")))
    (if (nil? (:label tree))
      (if (not (nil? (first (filter #((:function %) instance) (:edges tree)))))
        (dt-classify instance (:next-node (first (filter #((:function %) instance)
                                                         (:edges tree)))))
        (throw (Exception. (str tree))))
      (first (:label tree)))))

(defn rf-classify
  [instance forest]
  (->> forest
       (map #(dt-classify instance %))
       frequencies))

(def skew-constant 2)

(defn skew-conditional-entropy
  [x-attr instance-dict]
  (let [instances (keys instance-dict)
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
                            [x (filter x instances)]))
        y-splits (into {} (for [y y-filters]
                            [y (filter y instances)]))]
    [x-splits
     (core/sum
      (for [x x-filters]
        (* (core/sum
            (for [y y-filters]
              (let [fil-y (get y-splits y)
                    pygx (p-y-given-x (skew/sum-weights (filter #(x %) fil-y) instance-dict)
                                      (skew/sum-weights fil-y instance-dict)
                                      y-attr)]
                (core/entropy pygx))))
           (px (skew/sum-weights (get x-splits x) instance-dict)
               (skew/sum-weights instances instance-dict)
               x-attr)
           -1)))]))

(defn best-skew-split
  [instance-dict possible-features]
  (apply max-key #(second (second %)) (for [attr possible-features]
                                        [attr (skew-conditional-entropy attr instance-dict)])))

(defn majority-skew-class-vote
  [instance-dict instance-seq]
  (let [y-attr (. (first instance-seq) classAttribute)
        attr-seq (core/attribute-vals y-attr)
        total-weight (skew/sum-weights instance-seq instance-dict)]
    (apply max-key second (for [attr-val attr-seq]
                            [attr-val (/ (skew/sum-weights (filter #(= attr-val (. % stringValue y-attr))
                                                                   instance-seq)
                                                           instance-dict)
                                         total-weight)]))))

(defn dt-make-skew-subtree
  "data: training set, possibe-features: set of features that can be spit over"
  [node instance-dict possible-features]
  (let [instance-seq (keys instance-dict)]
    (if (stop-making-subtree? instance-seq possible-features 2)
      (-> node
          (add-label (majority-skew-class-vote instance-dict instance-seq))
          (add-data instance-seq))
      (let [best (best-skew-split instance-dict possible-features)
            best-attr (first best)
            conditional-entropy (second (second best))
            x-splits (first (second best))]
        (if (> 0 conditional-entropy)
          (-> node
              (add-label (majority-skew-class-vote instance-dict instance-seq))
              (add-data instance-seq))
          (reduce add-edge
                  (add-feature node best-attr)
                  (for [[val insts] x-splits]
                    (if (empty? insts)
                      (make-edge (:split (meta val))
                                 val
                                 (add-label (make-node nil) (majority-skew-class-vote instance-dict instance-seq)))
                      (make-edge (:split (meta val))
                                 val
                                 (dt-make-skew-subtree (make-node nil)
                                                       (into {} (for [inst insts] [inst (get instance-dict inst)]))
                                                       (disj possible-features best-attr)))))))))))

(defn build-skew-decision-tree
  [instances k skew-constant]
  (if (> (count instances) 0)
    (let [instances (seq instances)]
      (for [i (range k)]
        (dt-make-skew-subtree (make-node nil)
                              (skew/skew instances skew-constant)
                              (set (core/attribute-seq instances)))))
    (throw (Exception. "instances must have at least at least one element"))))

(defn dt-make-randomized-skew-subtree
  "data: training set, possible-features: set of features that can be split over"
  [node instance-dict possible-features]
  (let [instance-seq (keys instance-dict)]
    (if (stop-making-subtree? instance-seq possible-features 2)
      (-> node
          (add-label (majority-skew-class-vote instance-dict instance-seq))
          (add-data instance-seq))
      (let [best (best-skew-split instance-dict (take (+ (quot (count possible-features) 2) 1)
                                                      (simple/sample possible-features)))
            best-attr (first best)
            conditional-entropy (second (second best))
            x-splits (first (second best))]
        (if (> 0 conditional-entropy)
          (-> node
              (add-label (majority-skew-class-vote instance-dict instance-seq))
              (add-data instance-seq))
          (reduce add-edge
                  (add-feature node best-attr)
                  (for [[val insts] x-splits]
                    (if (empty? insts)
                      (make-edge (:split (meta val))
                                 val
                                 (add-label (make-node nil)
                                            (majority-skew-class-vote instance-dict instance-seq)))
                      (make-edge (:split (meta val))
                                 val
                                 (dt-make-randomized-skew-subtree (make-node nil)
                                                                  (into {} (for [inst insts] [inst (get instance-dict inst)]))
                                                                  (disj possible-features best-attr)))))))))))
(defn build-skew-random-forest
  "returns a list of t random decision trees"
  [instances t skew-constant]
  (for [i (range t)]
    (let [new-instances (core/sample-with-replacement instances (count instances))]
      (dt-make-randomized-skew-subtree (make-node nil)
                                       (skew/skew new-instances skew-constant)
                                       (set (core/attribute-seq new-instances))))))
(defn count-data
  [tree]
  (if (empty? (:edges tree))
    (count (:data tree))
    (+ (count (:data tree))
       (apply + (for [e (:edges tree)]
                  (count-data (:next-node e)))))))
"(defn check-complete
  \"checks the condition: either there's a prediction, or there's children \"
  [tree]
  (if (core/xor (empty? (:edges tree)) (nil? (:feature tree)))
    (apply #(and %) )))"
