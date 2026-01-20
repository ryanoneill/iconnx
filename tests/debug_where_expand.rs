use iconnx::attributes::NodeAttributes;
use iconnx::operators::expand::Expand;
use iconnx::operators::where_op::Where;
/// Debug the Where -> Expand issue at node 1251-1252
///
/// This test isolates the problematic nodes to understand the shape computation
use iconnx::tensor::Tensor;

#[test]
#[ignore]
fn test_debug_where_expand_shape_issue() {
    println!("\n🔬 Debugging Where -> Expand shape computation\n");

    // According to ORT:
    // - Where should output [3] with values [1, 3, 128]
    // - Expand should expand [1, 128] to [1, 3, 128]

    // But we're getting:
    // - Where outputs [3] with values [1, 3, 1]
    // - Expand tries to expand [1, 128] to [1, 3, 1] (fails!)

    // Test Where with sample inputs
    println!("Testing Where operator:");

    let condition = Tensor::from_vec(vec![1.0, 1.0, 0.0], vec![3]); // true, true, false
    let x_values = Tensor::from_vec_i64(vec![1, 3, 128], vec![3]); // [1, 3, 128]
    let y_values = Tensor::from_vec_i64(vec![9, 9, 9], vec![3]); // fallback values

    let where_result = Where::forward(
        &[condition, x_values.clone(), y_values],
        &NodeAttributes::new(),
    );

    println!("  condition: [true, true, false]");
    println!("  x: {:?}", x_values.as_slice_i64());
    println!("  y: [9, 9, 9]");
    println!("  result: {:?}", where_result.as_slice_i64());
    println!("  Expected: [1, 3, 128] (select from x)");

    assert_eq!(where_result.as_slice_i64(), &[1, 3, 128]);

    println!("\n✅ Where works correctly with Int64!\n");

    // Now test if the values are preserved through Expand
    let data_to_expand = Tensor::from_vec(vec![0.5; 128], vec![1, 128]);
    let expand_result = Expand::forward(&[data_to_expand, where_result], &NodeAttributes::new());

    println!("Testing Expand operator:");
    println!("  input: [1, 128]");
    println!("  target shape from Where: [1, 3, 128]");
    println!("  result.shape: {:?}", expand_result.shape());
    println!("  Expected: [1, 3, 128]");

    assert_eq!(expand_result.shape(), &[1, 3, 128]);

    println!("\n✅ Expand works correctly!");
}
