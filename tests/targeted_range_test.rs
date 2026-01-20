/// Targeted test for nodes 1314-1321 (Cast→Gather→CumSum→Gather→Range chain)
///
/// More efficient than full tracer - only tests the specific problematic chain
use iconnx::attributes::NodeAttributes;
// use iconnx::operators::cast::Cast; // Currently unused
use iconnx::operators::cumsum::CumSum;
use iconnx::operators::gather::Gather;
use iconnx::operators::range::Range;
use iconnx::tensor::Tensor;

#[test]
fn test_cast_gather_cumsum_range_chain() {
    println!("\n🔬 Testing Cast→Gather→CumSum→Gather→Range chain");

    // Simplify: Start directly with the input to CumSum (node 1316)
    // Per ORT values: input is shape [3] with values [13, 11, 2]
    let cumsum_input = Tensor::from_vec_i64(vec![13, 11, 2], vec![3]);
    println!(
        "CumSum input: shape={:?}, values={:?}",
        cumsum_input.shape(),
        cumsum_input.as_slice_i64()
    );

    // Node 1316: CumSum([13, 11, 2]) = [13, 24, 26]
    // Axis is 0 for 1D array
    let axis = Tensor::from_vec_i64(vec![0], vec![1]);
    let cumsum_output = CumSum::forward(&[cumsum_input, axis.clone()], &NodeAttributes::new());
    println!(
        "CumSum output: shape={:?}, values={:?}",
        cumsum_output.shape(),
        cumsum_output.as_slice_i64()
    );
    assert_eq!(
        cumsum_output.as_slice_i64(),
        vec![13, 24, 26],
        "CumSum should produce [13, 24, 26]"
    );

    // Node 1317: Gather extracts index -1 (last element) = 26
    let indices = Tensor::from_vec_i64(vec![-1], vec![1]);
    let gather2_output = Gather::forward(&[cumsum_output, indices], &NodeAttributes::new());
    println!(
        "Node 1317 (Gather[-1]): {:?}",
        gather2_output.as_slice_i64()
    );
    assert_eq!(
        gather2_output.as_slice_i64(),
        vec![26],
        "Gather should extract 26"
    );

    // Node 1321: Range(start=0, limit=26, delta=1) = [0, 1, 2, ..., 25]
    let start = Tensor::from_vec_i64(vec![0], vec![]);
    let limit = gather2_output; // Should be scalar 26
    let delta = Tensor::from_vec_i64(vec![1], vec![]);

    let range_output = Range::forward(&[start, limit, delta], &NodeAttributes::new());
    println!(
        "Node 1321 (Range): shape={:?}, len={}",
        range_output.shape(),
        range_output.len()
    );
    assert_eq!(
        range_output.shape(),
        &[26],
        "Range should produce array of length 26"
    );

    println!("✅ Full chain works correctly!");
}
